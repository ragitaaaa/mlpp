"""LightGBM custom point-process objectives for mlpp.

Provides a unified `make_objective` factory returning `(gradient, hessian)`
callables for:
1. Poisson (`poisson`)
2. Weighted Poisson (`weighted_poisson`)
3. Logistic (`logistic`)
4. Weighted Logistic (`weighted_logistic`)
5. Iterative Poisson (`iterative_poisson`)
6. Iterative Logistic (`iterative_logistic`)
"""

import numpy as np


_CLIP_MIN = -50.0
_CLIP_MAX = 50.0
_MIN_DENOM = 1e-4
_POISSON_OBJS = ("poisson", "weighted_poisson", "iterative_poisson")
_LOGISTIC_OBJS = ("logistic", "weighted_logistic", "iterative_logistic")
_ALL_OBJS = _POISSON_OBJS + _LOGISTIC_OBJS


def _clip_eta(preds):
    """Clip raw log-intensity predictions before exponentiation."""
    return np.clip(np.asarray(preds, dtype=float), _CLIP_MIN, _CLIP_MAX)


def _safe_sigmoid(eta, log_delta):
    """Compute numerically stable sigmoid(eta - log(delta))."""
    logit = np.clip(eta - log_delta, _CLIP_MIN, _CLIP_MAX)
    return 1.0 / (1.0 + np.exp(-logit))


def make_objective(objective, v=None, delta=None, w=None, A=None):
    """Create a custom point-process objective callable for LightGBM."""
    if objective not in _ALL_OBJS:
        raise ValueError(f"Unsupported objective: {objective}")

    is_poisson = objective in _POISSON_OBJS
    v_arr = np.asarray(v, dtype=float) if is_poisson else None
    if not is_poisson:
        delta_arr = np.asarray(delta, dtype=float)
        log_delta = np.log(delta_arr)
    else:
        delta_arr = None
        log_delta = None

    w_arr = np.asarray(w, dtype=float) if objective in ("weighted_poisson", "weighted_logistic") else None
    A_arr = np.asarray(A, dtype=float) if objective in ("iterative_poisson", "iterative_logistic") else None

    def objective_fn(preds, train_data):
        eta = _clip_eta(preds)
        y = np.asarray(train_data.get_label(), dtype=float)

        if is_poisson:
            lam = np.exp(eta)
            grad = v_arr * lam - y
            hess = v_arr * lam
            if objective == "weighted_poisson":
                grad = w_arr * grad
                hess = w_arr * hess
            elif objective == "iterative_poisson":
                wt = 1.0 / np.maximum(_MIN_DENOM, 1.0 + lam * A_arr)
                wt = wt / np.mean(wt)
                grad = wt * grad
                hess = wt * hess
        else:
            pi = _safe_sigmoid(eta, log_delta)
            grad = pi - y
            hess = pi * (1.0 - pi)
            if objective == "weighted_logistic":
                grad = w_arr * grad
                hess = w_arr * hess
            elif objective == "iterative_logistic":
                lam = np.exp(eta)
                denom = delta_arr * np.maximum(_MIN_DENOM, 1.0 + lam * A_arr)
                wt = (lam + delta_arr) / denom
                wt = wt / np.mean(wt)
                grad = wt * grad
                hess = wt * hess

        return grad, hess

    return objective_fn
