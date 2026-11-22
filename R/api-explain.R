#' Extract TreeSHAP values for a GBPP model
#'
#' @param object A `gbpp_model` object.
#' @param newdata A data frame of covariate values to explain.
#' @param ... Additional arguments.
#'
#' @keywords internal
#' @noRd
.gbpp_shap <- function(object, newdata, ...) {
  UseMethod(".gbpp_shap")
}

#' Extract Model Explanations (SHAP)
#'
#' Extracts exact tree-path-dependent Shapley Additive Explanations (TreeSHAP)
#' on the additive log-intensity (link) scale from a fitted GBPP model, including
#' the model baseline offset so that `rowSums(shap) + bias` equals
#' `predict(object, newdata, type = "link")`.
#'
#' Note that TreeSHAP computes tree-path conditional expectations over the
#' fitted splits rather than interventional expectations against an external
#' background sample; the `background` argument selects which subset of `data`
#' observations (`"quadrature"`, `"events"`, or `"all"`) is evaluated and returned.
#'
#' @param object A fitted `gbpp_model` object.
#' @param data A `gbpp_data` object whose observations are to be explained.
#' @param background Character string specifying which subset of `data` rows
#'   to evaluate: `"quadrature"` (default, dummy points `y == 0`), `"events"`
#'   (observed event points `y == 1`), or `"all"` (all rows).
#' @param ... Additional arguments passed to engine SHAP methods.
#'
#' @details
#' TreeSHAP decomposes each observation's predicted log-intensity \eqn{\hat{\eta}(u)}
#' into an intercept term `bias` plus exact feature attributions \eqn{\phi_j(u)}
#' satisfying
#' \deqn{\hat{\eta}(u) = \mathrm{bias}(u) + \sum_{j=1}^{p} \phi_j(u).}
#' Subsetting via `background` selects which rows of `data` are evaluated without
#' altering the tree-path conditioning embedded in the fitted trees.
#'
#' @return A list of class `"gbpp_explanation"` containing:
#'   \item{shap}{A numeric matrix of SHAP values for each feature on the additive link scale.}
#'   \item{bias}{A numeric vector of baseline plus expected tree contribution on the link scale.}
#'   \item{data}{The subsetted covariate data frame corresponding to the evaluated rows.}
#'
#' @references
#' Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B.,
#' Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020).
#' From local explanations to global understanding with explainable AI for trees.
#' *Nature Machine Intelligence*, 2(1), 56--67.
#' \doi{10.1038/s42256-019-0138-9}
#'
#' @seealso [fit_gbpp()], [print.gbpp_explanation()], [plot.gbpp_explanation()]
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
#'   exp_obj <- explain_gbpp(fit, dat, background = "events")
#'   print(exp_obj)
#' }
#' @export
explain_gbpp <- function(
  object,
  data,
  background = c("quadrature", "events", "all"),
  ...
) {
  if (!inherits(object, "gbpp_model")) {
    stop("`object` must be a fitted gbpp_model.", call. = FALSE)
  }
  if (!inherits(data, "gbpp_data")) {
    stop("`data` must be a gbpp_data object.", call. = FALSE)
  }

  background <- match.arg(background)

  X <- data$X
  y <- data$y

  baseline <- attr(object, "baseline")
  if (!is.null(baseline) && length(baseline) > 1L && length(baseline) != length(y)) {
    stop(
      "Model was fitted with a length-", length(baseline),
      " observation-specific `baseline` vector; `data` must have ",
      length(baseline), " observations.",
      call. = FALSE
    )
  }

  idx <- switch(background,
    quadrature = which(y == 0),
    events = which(y == 1),
    all = seq_len(nrow(X))
  )

  if (length(idx) == 0L) {
    stop("No points found for the requested background subset.", call. = FALSE)
  }

  X_bg <- X[idx, , drop = FALSE]

  res <- .gbpp_shap(object, X_bg, ...)

  if (!is.null(baseline) && length(baseline) == length(y) && length(idx) != length(y)) {
    # When subsetting rows from a vector-baseline fit, add the subsetted baseline
    if (length(res$bias) == length(idx)) {
      res$bias <- res$bias + as.numeric(baseline)[idx]
    }
  }

  res$data <- X_bg

  class(res) <- c("gbpp_explanation", class(res))
  res
}

#' Print TreeSHAP Explanation Summary
#'
#' Prints evaluation sample count, feature count, mean baseline bias, and global
#' feature importance ranked by mean absolute SHAP value on the log-intensity scale.
#'
#' @param x A `gbpp_explanation` object produced by [explain_gbpp()].
#' @param ... Additional arguments (currently unused).
#'
#' @details
#' Ranks predictors in descending order of
#' \eqn{n_{\mathrm{eval}}^{-1}\sum_{i=1}^{n_{\mathrm{eval}}} |\phi_{ij}|}, where
#' \eqn{n_{\mathrm{eval}}} is the number of evaluated rows in the chosen
#' `background` subset.
#'
#' @return Invisibly returns `x`.
#'
#' @seealso [explain_gbpp()], [plot.gbpp_explanation()]
#'
#' @examples
#' exp_demo <- structure(
#'   list(
#'     shap = matrix(c(0.2, -0.1, 0.05, 0.3), ncol = 2, dimnames = list(NULL, c("x", "y"))),
#'     bias = c(-1.0, -1.0),
#'     data = data.frame(x = c(0.2, 0.8), y = c(0.4, 0.7))
#'   ),
#'   class = "gbpp_explanation"
#' )
#' print(exp_demo)
#' @export
print.gbpp_explanation <- function(x, ...) {
  cat("TreeSHAP Model Explanation\n")
  cat("--------------------------\n")
  cat("Evaluation samples :", nrow(x$shap), "\n")
  cat("Number of features :", ncol(x$shap), "\n")
  cat("Baseline bias      :", round(mean(x$bias), 4), "\n")
  cat("\nMean absolute SHAP values (Feature Importance):\n")
  mean_abs <- colMeans(abs(x$shap))
  mean_abs <- sort(mean_abs, decreasing = TRUE)
  for (i in seq_along(mean_abs)) {
    cat(sprintf("  %-16s: %.4f\n", names(mean_abs)[i], mean_abs[i]))
  }
  invisible(x)
}

#' Plot TreeSHAP Explanations
#'
#' Visualizes feature importance barplots or partial dependence scatter plots
#' derived from TreeSHAP values.
#'
#' @param x A `gbpp_explanation` object.
#' @param type Type of plot: `"importance"` for global mean absolute SHAP barplot, or
#'   `"dependence"` for feature dependence curve. Default is `"importance"`.
#' @param feature Name of feature to plot when `type = "dependence"`. Defaults to
#'   the most important feature.
#' @param ... Additional arguments passed to base plotting functions.
#'
#' @details
#' When `type = "importance"`, draws a horizontal barplot of mean absolute SHAP
#' values across all evaluated observations. When `type = "dependence"`, plots
#' the covariate values for `feature` (either continuous scatter or categorical
#' boxplots) against their corresponding SHAP contributions on the log-intensity scale.
#'
#' @return Invisibly returns `x`.
#'
#' @seealso [explain_gbpp()], [print.gbpp_explanation()]
#'
#' @examples
#' exp_demo <- structure(
#'   list(
#'     shap = matrix(c(0.2, -0.1, 0.05, 0.3), ncol = 2, dimnames = list(NULL, c("x", "y"))),
#'     bias = c(-1.0, -1.0),
#'     data = data.frame(x = c(0.2, 0.8), y = c(0.4, 0.7))
#'   ),
#'   class = "gbpp_explanation"
#' )
#' plot(exp_demo, type = "importance")
#' plot(exp_demo, type = "dependence", feature = "x")
#' @export
plot.gbpp_explanation <- function(x, type = c("importance", "dependence"), feature = NULL, ...) {
  type <- match.arg(type)

  if (type == "importance") {
    imp <- colMeans(abs(x$shap))
    imp <- sort(imp, decreasing = FALSE)
    graphics::barplot(
      imp,
      horiz = TRUE,
      las = 1,
      col = "steelblue",
      border = NA,
      xlab = "Mean |SHAP value|",
      main = "Feature Importance (TreeSHAP)",
      ...
    )
  } else if (type == "dependence") {
    if (is.null(feature)) {
      imp <- colMeans(abs(x$shap))
      feature <- names(sort(imp, decreasing = TRUE))[1]
    }
    if (!feature %in% colnames(x$shap)) {
      stop(sprintf("Feature '%s' not found in explanation data.", feature), call. = FALSE)
    }

    feat_x <- x$data[[feature]]
    shap_y <- x$shap[, feature]

    if (is.factor(feat_x) || is.character(feat_x)) {
      graphics::boxplot(
        shap_y ~ feat_x,
        xlab = feature,
        ylab = paste("SHAP value for", feature),
        col = "lightblue",
        main = paste("TreeSHAP Dependence:", feature),
        ...
      )
    } else {
      graphics::plot(
        feat_x,
        shap_y,
        xlab = feature,
        ylab = paste("SHAP value for", feature),
        pch = 16,
        col = grDevices::rgb(0.2, 0.4, 0.8, 0.4),
        main = paste("TreeSHAP Dependence:", feature),
        ...
      )
      if (length(unique(feat_x)) > 5) {
        fit_smooth <- stats::lowess(feat_x, shap_y)
        graphics::lines(fit_smooth, col = "red", lwd = 2)
      }
    }
  }

  invisible(x)
}
