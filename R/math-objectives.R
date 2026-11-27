# ============================================================
# Internal Mathematical Helpers: Dummy Intensity, Iterative
# Weights, and Pure-R Reference Objective Derivatives
# ============================================================

#' Extract or calculate dummy-process intensity (`delta`)
#'
#' When `x` is a `spatstat` quadrature scheme (`quad`), extracts and validates
#' `x$param$rho`. Otherwise treats `x` as a binary response vector `y` (`0`/`1`)
#' and computes `sum(y == 0) / domain_measure` from an `owin`, `linnet`, or
#' positive quadrature weight vector `measure`.
#'
#' @keywords internal
#' @noRd
.dummy_intensity <- function(x, measure = NULL) {
  if (inherits(x, "quad")) {
    n <- length(spatstat.geom::is.data(x))
    delta <- x$param$rho
    if (is.null(delta)) {
      stop(
        paste0(
          "Dummy-process intensity could not be extracted from `Q`. ",
          "Provide it explicitly for quadrature schemes that do not ",
          "store `rho` in `Q$param$rho`."
        ),
        call. = FALSE
      )
    }
    if (!is.numeric(delta)) {
      stop("`Q$param$rho` must be numeric.", call. = FALSE)
    }
    if (any(!is.finite(delta)) || any(delta <= 0)) {
      stop("Dummy-process intensity must contain positive finite values.", call. = FALSE)
    }
    if (length(delta) == 1L) {
      delta <- rep(delta, n)
    }
    if (length(delta) != n) {
      stop(
        "Dummy-process intensity must either be a single value or have one value for each quadrature point.",
        call. = FALSE
      )
    }
    return(delta)
  }

  y <- x
  if (!is.numeric(y) ||
    length(y) == 0L ||
    anyNA(y) ||
    any(!is.finite(y)) ||
    any(!y %in% c(0, 1))) {
    stop("`y` must be a non-empty finite binary numeric vector.", call. = FALSE)
  }

  n_dummy <- sum(y == 0)
  if (n_dummy <= 0) {
    stop("At least one dummy point is required.", call. = FALSE)
  }

  domain_measure <- if (inherits(measure, "owin")) {
    spatstat.geom::area(measure)
  } else if (inherits(measure, "linnet")) {
    spatstat.geom::volume(measure)
  } else {
    if (!is.numeric(measure) ||
      !(length(measure) == length(y) || length(measure) == 1L) ||
      anyNA(measure) ||
      any(!is.finite(measure)) ||
      any(measure <= 0)) {
      stop(
        "`measure` must be a positive finite numeric vector with the same length as `y` (or length 1), an `owin`, or a `linnet`.",
        call. = FALSE
      )
    }
    sum(measure)
  }

  if (!is.finite(domain_measure) || domain_measure <= 0) {
    stop("The total domain measure must be positive and finite.", call. = FALSE)
  }

  delta <- n_dummy / domain_measure
  if (!is.finite(delta) || delta <= 0) {
    stop("Failed to calculate a positive finite dummy intensity.", call. = FALSE)
  }

  delta
}

#' Calculate iteratively updated Poisson or Logistic dependence weights
#'
#' Computes unit-mean normalized dependence weights for `"iterative_poisson"`
#' (`delta = NULL`) or `"iterative_logistic"` (`delta` supplied).
#'
#' @keywords internal
#' @noRd
.iter_weight <- function(
  eta,
  A,
  delta = NULL,
  clamp = FALSE,
  min_denom = 1e-4
) {
  if (!is.numeric(eta) || !is.numeric(A)) {
    stop("`eta` and `A` must be numeric.", call. = FALSE)
  }
  if (length(eta) != length(A)) {
    stop("`eta` and `A` must have the same length.", call. = FALSE)
  }
  if (any(!is.finite(eta)) || any(!is.finite(A))) {
    stop("`eta` and `A` must contain only finite values.", call. = FALSE)
  }

  if (!is.null(delta)) {
    if (!is.numeric(delta) || length(delta) != length(eta)) {
      stop("`delta` must be a numeric vector with the same length as `eta`.", call. = FALSE)
    }
    if (any(!is.finite(delta)) || any(delta <= 0)) {
      stop("`delta` must contain positive finite values.", call. = FALSE)
    }
  }

  eta_c <- pmin(pmax(eta, -50), 50)
  lam <- exp(eta_c)
  denom <- 1 + lam * A

  if (isTRUE(clamp)) {
    denom <- pmax(min_denom, denom)
  } else if (any(denom <= 0)) {
    stop(
      "The iterative weight is undefined for at least one observation because `1 + lambda * A` is not positive.",
      call. = FALSE
    )
  }

  w <- if (is.null(delta)) {
    1 / denom
  } else {
    (lam + delta) / (delta * denom)
  }

  w / mean(w)
}

#' Pure-R reference derivatives (gradient and Fisher scoring curvature)
#'
#' Evaluates `gradient` and `hessian` of the 6 point process objectives on the
#' log-intensity scale (`eta_i`). Used for verification and unit tests.
#'
#' @keywords internal
#' @noRd
.ref_deriv <- function(
  data,
  eta,
  objective = c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  ),
  clamp = TRUE
) {
  objective <- match.arg(objective)
  .check_data(data, objective = objective)

  if (!is.numeric(eta) || length(eta) != length(data$y) || any(!is.finite(eta))) {
    stop("`eta` must be a finite numeric vector with one value per observation.", call. = FALSE)
  }

  eta_c <- pmin(pmax(eta, -50), 50)
  is_poisson <- objective %in% c("poisson", "weighted_poisson", "iterative_poisson")

  if (is_poisson) {
    lam <- exp(eta_c)
    g0 <- data$quadrature_weight * lam - data$y
    h0 <- data$quadrature_weight * lam
  } else {
    pi <- stats::plogis(eta_c - log(data$dummy_intensity))
    g0 <- pi - data$y
    h0 <- pi * (1 - pi)
  }

  if (objective %in% c("poisson", "logistic")) {
    return(list(gradient = g0, hessian = h0))
  }

  if (objective %in% c("weighted_poisson", "weighted_logistic")) {
    w <- data$statistical_weight
    return(list(gradient = w * g0, hessian = w * h0))
  }

  w <- .iter_weight(
    eta = eta_c,
    A = data$interaction_integral,
    delta = if (is_poisson) NULL else data$dummy_intensity,
    clamp = clamp
  )
  list(gradient = w * g0, hessian = w * h0, weight = w)
}
