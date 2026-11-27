"""CatBoost custom point-process objectives for mlpp."""

import math

import numpy as np


_CLIP_MIN = -50.0
_CLIP_MAX = 50.0
_MIN_DENOM = 1e-4


def _as_float_array(values, name):
    """Convert values to a finite one-dimensional float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")

    return arr


def _check_length(values, n, name):
    """Check that an array has the expected length."""
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}, got {len(values)}.")


def _get_weights(weights, n):
    """Return validated CatBoost Pool weights or unit weights when omitted."""
    if weights is None:
        return np.ones(n, dtype=float)

    weights = _as_float_array(weights, "weights")
    _check_length(weights, n, "weights")

    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")

    return weights


class PoissonLoss:
    """Stateless, Numba-compilable Poisson / weighted Poisson objective.

    Caller supplies `baseline = eta0 + log(quadrature_weight)` and optional
    `weight = statistical_weight` via `catboost.Pool`. Keeping the pure-Python
    scalar loop enables CatBoost Numba JIT compilation and exact execution
    under `Ordered` boosting, row subsampling, and multi-threading.
    """

    def calc_ders_range(self, approxes, targets, weights):
        """Return (-gradient, -hessian) pairs for Poisson point process."""
        result = []
        for i in range(len(approxes)):
            a = approxes[i]
            if a < -50.0:
                a = -50.0
            elif a > 50.0:
                a = 50.0
            ea = math.exp(a)
            w = 1.0 if weights is None else weights[i]
            result.append((w * (targets[i] - ea), -w * ea))
        return result


class LogisticLoss:
    """Stateless, Numba-compilable Logistic / weighted Logistic objective.

    Caller supplies `baseline = eta0 - log(dummy_intensity)` and optional
    `weight = statistical_weight` via `catboost.Pool`.
    """

    def calc_ders_range(self, approxes, targets, weights):
        """Return (-gradient, -hessian) pairs for Baddeley logistic regression."""
        result = []
        for i in range(len(approxes)):
            a = approxes[i]
            if a < -50.0:
                a = -50.0
            elif a > 50.0:
                a = 50.0
            pi = 1.0 / (1.0 + math.exp(-a))
            w = 1.0 if weights is None else weights[i]
            result.append((w * (targets[i] - pi), -w * pi * (1.0 - pi)))
        return result


class _BatchObjective:
    """Manage sequential observation slices supplied by CatBoost."""

    def __init__(self, n_obs):
        self._n_obs = int(n_obs)
        self._offset = 0

    def _next_batch(self, n):
        """Return the starting index of the next sequential CatBoost batch."""
        n = int(n)
        if n < 0:
            raise ValueError("Batch size must be non-negative.")

        if self._offset == self._n_obs:
            self._offset = 0

        if self._offset + n > self._n_obs:
            raise ValueError(
                "CatBoost batch exceeds the stored observation range: "
                f"offset={self._offset}, batch_size={n}, n_obs={self._n_obs}."
            )

        start = self._offset
        self._offset += n
        return start

    def _batch_slice(self, values, start, n, name):
        """Return the observation-level values for the current batch."""
        _check_length(values, self._n_obs, name)
        return values[start:start + n]


class IterPoissonLoss(_BatchObjective):
    """Iteratively reweighted Poisson objective for CatBoost (`Plain` mode)."""

    def __init__(self, v, A):
        self.v = _as_float_array(v, "quadrature_weight")
        self.A = _as_float_array(A, "interaction_integral")
        _check_length(self.A, len(self.v), "interaction_integral")
        super().__init__(len(self.v))

    def calc_ders_range(self, approxes, targets, weights):
        """Return (-gradient, -hessian) pairs with dynamic dependence weights."""
        targets = _as_float_array(targets, "targets")
        approxes = _as_float_array(approxes, "approxes")
        n = len(targets)
        _check_length(approxes, n, "approxes")

        start = self._next_batch(n)
        v = self._batch_slice(self.v, start, n, "quadrature_weight")
        A = self._batch_slice(self.A, start, n, "interaction_integral")
        cb_w = _get_weights(weights, n)

        eta = np.clip(approxes, _CLIP_MIN, _CLIP_MAX)
        lam = np.exp(eta)
        w = 1.0 / np.maximum(_MIN_DENOM, 1.0 + lam * A)
        mean_w = np.mean(w)
        if mean_w > 0.0:
            w = w / mean_w

        gradient = cb_w * w * (v * lam - targets)
        hessian = cb_w * w * v * lam
        return list(zip((-gradient).tolist(), (-hessian).tolist()))


class IterLogisticLoss(_BatchObjective):
    """Iteratively reweighted logistic objective for CatBoost (`Plain` mode)."""

    def __init__(self, delta, A):
        self.delta = _as_float_array(delta, "dummy_intensity")
        if np.any(self.delta <= 0.0):
            raise ValueError("dummy_intensity must be strictly positive.")
        self.log_delta = np.log(self.delta)

        self.A = _as_float_array(A, "interaction_integral")
        _check_length(self.A, len(self.delta), "interaction_integral")
        super().__init__(len(self.delta))

    def calc_ders_range(self, approxes, targets, weights):
        """Return (-gradient, -hessian) pairs with dynamic dependence weights."""
        targets = _as_float_array(targets, "targets")
        approxes = _as_float_array(approxes, "approxes")
        n = len(targets)
        _check_length(approxes, n, "approxes")

        start = self._next_batch(n)
        delta = self._batch_slice(self.delta, start, n, "dummy_intensity")
        log_delta = self._batch_slice(self.log_delta, start, n, "log_dummy_intensity")
        A = self._batch_slice(self.A, start, n, "interaction_integral")
        cb_w = _get_weights(weights, n)

        eta = np.clip(approxes, _CLIP_MIN, _CLIP_MAX)
        lam = np.exp(eta)
        w = (lam + delta) / (delta * np.maximum(_MIN_DENOM, 1.0 + lam * A))
        mean_w = np.mean(w)
        if mean_w > 0.0:
            w = w / mean_w

        logit = np.clip(eta - log_delta, _CLIP_MIN, _CLIP_MAX)
        pi = 1.0 / (1.0 + np.exp(-logit))

        gradient = cb_w * w * (pi - targets)
        hessian = cb_w * w * pi * (1.0 - pi)
        return list(zip((-gradient).tolist(), (-hessian).tolist()))


def make_objective(objective, v=None, delta=None, w=None, A=None):
    """Create a CatBoost custom objective instance."""
    if objective in ("poisson", "weighted_poisson"):
        return PoissonLoss()
    if objective in ("logistic", "weighted_logistic"):
        return LogisticLoss()
    if objective == "iterative_poisson":
        return IterPoissonLoss(v=v, A=A)
    if objective == "iterative_logistic":
        return IterLogisticLoss(delta=delta, A=A)
    raise ValueError(f"Unsupported objective: {objective}")
