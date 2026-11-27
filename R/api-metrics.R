#' Evaluate Spatial Point Process Metrics
#'
#' Calculates the inhomogeneous Poisson point process log-likelihood and
#' Berman-Turner Poisson deviance, accepting either a fitted `mlpp_model` with
#' an `mlpp_data` object or raw prediction, response, and quadrature-weight
#' vectors.
#'
#' @param x Either a fitted `mlpp_model` object (evaluated on `data`), or a
#'   numeric vector of predicted intensities \eqn{\hat{\lambda}_i}.
#' @param data When `x` is an `mlpp_model`, an `mlpp_data` object with
#'   `quadrature_weight`. When `x` is a numeric intensity vector, the binary
#'   response vector `y` (`1` for events, `0` for dummy points).
#' @param quadrature_weight Numeric vector of Berman-Turner quadrature weights
#'   \eqn{v_i} when `x` is a numeric intensity vector. Ignored when `x` is an
#'   `mlpp_model`.
#' @param metric Which metric to return: `"all"` (default, returns named vector
#'   `c(loglik = ..., deviance = ...)`), `"loglik"` (scalar), or `"deviance"`
#'   (scalar).
#' @param p_scale Positive numeric thinning retention probability used when
#'   evaluating thinned test sets (e.g., `1 / folds` in [cv_mlpp()]). Default
#'   is `1`.
#'
#' @details
#' Approximates the inhomogeneous Poisson point process log-likelihood and
#' Berman-Turner GLM deviance over \eqn{N = n + m} quadrature points with
#' weights \eqn{v_i}:
#' \deqn{
#' \ell(\hat{\lambda}) = \sum_{i:\, y_i = 1} \log(p_{\mathrm{scale}}\,\hat{\lambda}_i)
#'   - \sum_{i=1}^{N} v_i \, p_{\mathrm{scale}} \, \hat{\lambda}_i,
#' }
#' \deqn{
#' D(\hat{\lambda}) = 2 \left(
#'   -\sum_{i:\, y_i = 1} \log(\mu_i) - \sum_{i=1}^{N} (y_i - \mu_i)
#' \right), \qquad \mu_i = v_i \, p_{\mathrm{scale}} \, \hat{\lambda}_i.
#' }
#' Positive floors (`1e-15`) are applied to scaled predicted intensities and
#' event quadrature weights `v[y == 1]` to guard against undefined logarithms.
#'
#' @return A named numeric vector `c(loglik = ..., deviance = ...)` when
#'   `metric = "all"`, or a numeric scalar when `metric` is `"loglik"` or
#'   `"deviance"`.
#'
#' @references
#' Berman, M., & Turner, T. R. (1992). Approximating point process likelihoods
#' with GLIM. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*,
#' 41(1), 31--38. \doi{10.2307/2347614}
#'
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' Cronie, O., Moradi, M., & Biscio, C. A. N. (2024). A cross-validation-based
#' statistical theory for point processes. *Biometrika*, 111(2), 625--641.
#' \doi{10.1093/biomet/asad041}
#'
#' @seealso [cv_mlpp()], [fit_gbpp()], [predict.mlpp_model()]
#'
#' @examples
#' y <- c(1, 0, 0)
#' v <- c(0.1, 0.45, 0.45)
#' lam <- c(2.0, 1.0, 1.0)
#' eval_mlpp(lam, y, v)
#' eval_mlpp(lam, y, v, metric = "deviance", p_scale = 0.5)
#'
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   eval_mlpp(fit, dat)
#' }
#' @export
eval_mlpp <- function(
  x,
  data = NULL,
  quadrature_weight = NULL,
  metric = c("all", "loglik", "deviance"),
  p_scale = 1
) {
  metric <- match.arg(metric)

  if (inherits(x, "mlpp_model")) {
    if (!inherits(data, "mlpp_data")) {
      stop("`data` must be an `mlpp_data` object when `x` is an `mlpp_model`.", call. = FALSE)
    }
    if (is.null(data$quadrature_weight)) {
      stop("`data$quadrature_weight` is required for `eval_mlpp()`.", call. = FALSE)
    }
    pred_intensity <- predict(x, newdata = data$X, type = "intensity")
    y <- data$y
    v <- data$quadrature_weight
  } else if (is.numeric(x)) {
    pred_intensity <- x
    y <- data
    v <- quadrature_weight
  } else {
    stop("`x` must be a fitted `mlpp_model` or a numeric vector of predicted intensities.", call. = FALSE)
  }

  if (!is.numeric(pred_intensity) || any(!is.finite(pred_intensity))) {
    stop("`pred_intensity` must be a finite numeric vector.", call. = FALSE)
  }
  if (!is.numeric(y) || any(!is.finite(y))) {
    stop("`y` must be a finite numeric vector.", call. = FALSE)
  }
  if (!all(y %in% c(0, 1))) {
    stop("`y` must contain only 0 (dummy) and 1 (event).", call. = FALSE)
  }
  if (!is.numeric(v) || any(!is.finite(v)) || any(v < 0)) {
    stop("`quadrature_weight` must be a finite non-negative numeric vector.", call. = FALSE)
  }
  n <- length(y)
  if (length(pred_intensity) != n || length(v) != n) {
    stop("`pred_intensity`, `y`, and `quadrature_weight` must have the same length.", call. = FALSE)
  }
  if (!is.numeric(p_scale) || length(p_scale) != 1L || !is.finite(p_scale) || p_scale <= 0) {
    stop("`p_scale` must be a single positive finite number.", call. = FALSE)
  }

  if (p_scale != 1) {
    pred_intensity <- pred_intensity * p_scale
  }
  pred_intensity <- pmax(pred_intensity, 1e-15)

  ll <- NULL
  if (metric %in% c("all", "loglik")) {
    ll <- sum(log(pred_intensity[y == 1])) - sum(v * pred_intensity)
  }

  dev <- NULL
  if (metric %in% c("all", "deviance")) {
    v_safe <- v
    v_safe[y == 1] <- pmax(v_safe[y == 1], 1e-15)
    mu <- v_safe * pred_intensity
    dev <- 2 * (sum(-log(mu[y == 1])) - sum(y - mu))
  }

  switch(metric,
    all = c(loglik = ll, deviance = dev),
    loglik = ll,
    deviance = dev
  )
}
