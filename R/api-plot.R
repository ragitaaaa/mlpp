#' Plot continuous spatial predictions from a fitted GBPP model
#'
#' Generates a spatial intensity or link scale map using the `spatstat` plotting engine.
#'
#' @param x A `gbpp_model` object fitted via [fit_gbpp()].
#' @param newdata A list of spatial covariates (`im` or `linim`) or a spatial
#'   window (`owin`) / linear network (`linnet`).
#' @param type Type of prediction to plot: `"intensity"` (default) or `"link"`.
#' @param overlay_points Logical; if `TRUE`, overlays `pts` on the prediction map.
#' @param pts A point pattern (`ppp` or `lpp`) to overlay when `overlay_points = TRUE`.
#' @param eps Spatial resolution (grid spacing or linear network sample spacing).
#' @param dimyx Spatial grid dimensions for planar `im` predictions.
#' @param ... Additional graphical arguments passed to `spatstat.geom::plot.im`
#'   or `spatstat.linnet::plot.linim`.
#'
#' @details
#' Internally delegates spatial grid or network mapping to [predict.gbpp_model()]
#' with `type = "im"` (for planar windows/images) or `type = "linim"` (for linear
#' networks/network images), then renders the resulting pixel or linear image
#' using `spatstat` plotting methods.
#'
#' @return Invisibly returns the plotted `spatstat.geom::im` or
#'   `spatstat.linnet::linim` object.
#'
#' @references
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' @seealso [fit_gbpp()], [predict.gbpp_model()], [explain_gbpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'     reticulate::py_available() &&
#'     reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_gbpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   pred_im <- plot(fit, newdata = list(x = cov_x), dimyx = c(16, 16))
#' }
#' @export
plot.gbpp_model <- function(
  x,
  newdata,
  type = c("intensity", "link"),
  overlay_points = FALSE,
  pts = NULL,
  eps = NULL,
  dimyx = NULL,
  ...
) {
  type <- match.arg(type)

  if (missing(newdata)) {
    stop("`newdata` must be supplied to generate a continuous spatial plot.", call. = FALSE)
  }

  is_network <- inherits(newdata, "linnet") ||
    (is.list(newdata) && length(newdata) > 0 &&
      all(vapply(newdata, inherits, logical(1), what = c("linim", "linfun"))))

  # Predict the spatial surface
  spatial_obj <- if (is_network) {
    predict(x, newdata = newdata, type = "linim", eps = eps)
  } else {
    predict(x, newdata = newdata, type = "im", eps = eps, dimyx = dimyx)
  }

  # Map to link scale if requested
  if (type == "link") {
    # spatstat supports Math.im and Math.linim generics, so log() works natively
    spatial_obj <- log(spatial_obj)
  }

  graphics::plot(spatial_obj, ...)

  if (overlay_points) {
    if (is.null(pts)) {
      warning("`pts` must be provided to overlay points.", call. = FALSE)
    } else {
      graphics::plot(pts, add = TRUE, col = "black", pch = 16, cex = 0.5)
    }
  }

  invisible(spatial_obj)
}
