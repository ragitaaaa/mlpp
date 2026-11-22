#' Spatial Poisson Log-Likelihood
#'
#' Calculates the Poisson point process log-likelihood using the Berman-Turner
#' quadrature approximation.
#'
#' @param pred_intensity Numeric vector of predicted intensity values.
#' @param y Binary numeric or integer vector indicating events (1) and dummy points (0).
#' @param w Numeric vector of quadrature weights.
#' @param p_scale Numeric scalar representing the thinning retention probability
#'   for test sets. Default is `1`.
#'
#' @details
#' Approximates the inhomogeneous Poisson point process log-likelihood on
#' planar window \eqn{W} (or linear network \eqn{\mathcal{L}}) via
#' Berman-Turner quadrature over \eqn{N = n + m} quadrature points with weights
#' \eqn{v_i} (supplied via `w`):
#' \deqn{
#' \ell(\hat{\lambda}) = \sum_{i:\, y_i = 1} \log(p_{\mathrm{scale}}\,\hat{\lambda}_i)
#'   - \sum_{i=1}^{N} v_i \, p_{\mathrm{scale}} \, \hat{\lambda}_i.
#' }
#' Predicted intensities are bounded below by `1e-15` prior to taking logarithms.
#' When evaluating held-out thinning folds from [cv_gbpp()], set `p_scale` to the
#' test retention probability \eqn{1/V}.
#'
#' @return Numeric scalar giving the evaluated spatial Poisson log-likelihood.
#'
#' @references
#' Berman, M., & Turner, T. R. (1992). Approximating point process likelihoods
#' with GLIM. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*,
#' 41(1), 31--38. \doi{10.2307/2347614}
#'
#' Cronie, O., Moradi, M., & Biscio, C. A. N. (2024). A cross-validation-based
#' statistical theory for point processes. *Biometrika*, 111(2), 625--641.
#' \doi{10.1093/biomet/asad041}
#'
#' @seealso [calc_poisson_deviance()], [cv_gbpp()]
#'
#' @examples
#' y <- c(1, 0, 0)
#' w <- c(0.1, 0.45, 0.45)
#' lam <- c(2.0, 1.0, 1.0)
#' calc_poisson_loglik(lam, y, w)
#' @export
calc_poisson_loglik <- function(pred_intensity, y, w, p_scale = 1) {
  .gbpp_validate_metric_inputs(pred_intensity, y, w, p_scale)

  if (p_scale != 1) {
    pred_intensity <- pred_intensity * p_scale
  }

  # Protect against log(0)
  pred_intensity <- pmax(pred_intensity, 1e-15)

  ll_events <- sum(log(pred_intensity[y == 1]))
  ll_integral <- sum(w * pred_intensity)

  ll_events - ll_integral
}

#' Spatial Poisson Deviance
#'
#' Calculates the Poisson point process deviance using the Berman-Turner quadrature approximation.
#'
#' @param pred_intensity Numeric vector of predicted intensity values.
#' @param y Binary numeric or integer vector indicating events (1) and dummy points (0).
#' @param w Numeric vector of quadrature weights (\eqn{v_i}).
#' @param p_scale Numeric scalar representing the thinning retention probability
#'   for test sets. Default is `1`.
#'
#' @details
#' Computes the Berman-Turner GLM Poisson deviance analog for point processes
#' over \eqn{N = n + m} quadrature points with weights \eqn{v_i} (supplied via
#' `w`):
#' \deqn{
#' D(\hat{\lambda}) = 2 \left(
#'   -\sum_{i:\, y_i = 1} \log(\mu_i) - \sum_{i=1}^{N} (y_i - \mu_i)
#' \right),
#' }
#' where \eqn{\mu_i = v_i \, p_{\mathrm{scale}} \, \hat{\lambda}_i}, with
#' positive floors applied to `pred_intensity` and event weights `w[y == 1]` to
#' guard against undefined logarithms.
#'
#' @return Numeric scalar giving the evaluated spatial Poisson deviance.
#'
#' @references
#' Berman, M., & Turner, T. R. (1992). Approximating point process likelihoods
#' with GLIM. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*,
#' 41(1), 31--38. \doi{10.2307/2347614}
#'
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' @seealso [calc_poisson_loglik()], [cv_gbpp()]
#'
#' @examples
#' y <- c(1, 0, 0)
#' w <- c(0.1, 0.45, 0.45)
#' lam <- c(2.0, 1.0, 1.0)
#' calc_poisson_deviance(lam, y, w)
#' @export
calc_poisson_deviance <- function(pred_intensity, y, w, p_scale = 1) {
  .gbpp_validate_metric_inputs(pred_intensity, y, w, p_scale)

  if (p_scale != 1) {
    pred_intensity <- pred_intensity * p_scale
  }

  pred_intensity <- pmax(pred_intensity, 1e-15)

  # Protect against exact 0 weights for events to prevent infinite deviance
  w_safe <- w
  w_safe[y == 1] <- pmax(w_safe[y == 1], 1e-15)

  mu <- w_safe * pred_intensity

  # GLM Poisson deviance: 2 * sum(y * log(y / mu) - (y - mu))
  # When y = 0, y * log(y / mu) = 0
  # When y = 1, y * log(y / mu) = -log(mu)

  dev_events <- sum(-log(mu[y == 1]))
  dev_all <- sum(y - mu)

  2 * (dev_events - dev_all)
}

#' Validate inputs for spatial Poisson metrics
#'
#' @keywords internal
#' @noRd
.gbpp_validate_metric_inputs <- function(pred_intensity, y, w, p_scale) {
  if (!is.numeric(pred_intensity) || any(!is.finite(pred_intensity))) {
    stop("`pred_intensity` must be a finite numeric vector.", call. = FALSE)
  }
  if (!is.numeric(y) || any(!is.finite(y))) {
    stop("`y` must be a finite numeric vector.", call. = FALSE)
  }
  if (!all(y %in% c(0, 1))) {
    stop("`y` must contain only 0 (dummy) and 1 (event).", call. = FALSE)
  }
  if (!is.numeric(w) || any(!is.finite(w)) || any(w < 0)) {
    stop("`w` must be a finite non-negative numeric vector.", call. = FALSE)
  }
  n <- length(y)
  if (length(pred_intensity) != n || length(w) != n) {
    stop("`pred_intensity`, `y`, and `w` must have the same length.", call. = FALSE)
  }
  if (!is.numeric(p_scale) || length(p_scale) != 1L || !is.finite(p_scale) || p_scale <= 0) {
    stop("`p_scale` must be a single positive finite number.", call. = FALSE)
  }
  invisible(TRUE)
}
