#' Extract TreeSHAP values for an mlpp model
#'
#' @param object An `mlpp_model` object.
#' @param newdata A data frame of covariate values to explain.
#' @param ... Additional arguments.
#'
#' @keywords internal
#' @noRd
.get_shap <- function(object, newdata, ...) {
  engine <- attr(object, "engine")
  if (is.null(engine)) {
    engine <- if (inherits(object, "gbpp_xgboost")) {
      "xgboost"
    } else if (inherits(object, "gbpp_lightgbm")) {
      "lightgbm"
    } else if (inherits(object, "gbpp_catboost")) {
      "catboost"
    } else {
      stop("Unknown mlpp model engine for TreeSHAP extraction.", call. = FALSE)
    }
  }

  object <- .unpack_model(object)
  feat_names <- attr(object, "feature_names")
  factor_levels <- attr(object, "factor_levels")
  n_rows <- nrow(newdata)

  shap_vals <- switch(engine,
    xgboost = {
      dmat <- .xgb_dmatrix(
        X = newdata,
        base_margin = rep(0.0, n_rows),
        factor_levels = factor_levels,
        feature_names = feat_names
      )
      object$predict(dmat, pred_contribs = TRUE)
    },
    lightgbm = {
      X_data <- .encode_X(
        newdata,
        factor_levels = factor_levels,
        feature_names = feat_names,
        engine = "lightgbm"
      )
      object$predict(as.matrix(X_data), pred_contrib = TRUE)
    },
    catboost = {
      .init_python()
      catboost <- reticulate::import("catboost", delay_load = FALSE)
      X_enc <- .encode_X(
        newdata,
        factor_levels = factor_levels,
        feature_names = feat_names,
        engine = "catboost"
      )
      cat_idx <- attr(object, "cat_features")
      if (is.null(cat_idx) && !is.null(factor_levels) && !is.null(feat_names)) {
        cat_idx <- as.integer(which(feat_names %in% names(factor_levels)) - 1L)
      }
      pool <- if (is.data.frame(X_enc) && !is.null(cat_idx) && length(cat_idx) > 0L) {
        catboost$Pool(data = X_enc, cat_features = as.list(as.integer(cat_idx)))
      } else {
        catboost$Pool(data = as.matrix(X_enc))
      }
      object$get_feature_importance(data = pool, type = "ShapValues")
    },
    stop("Unsupported engine for TreeSHAP extraction: ", engine, call. = FALSE)
  )

  if (is.null(feat_names)) {
    feat_names <- paste0("V", seq_len(ncol(shap_vals) - 1L))
  }
  colnames(shap_vals) <- c(feat_names, "bias")

  baseline <- attr(object, "baseline")
  bias <- as.numeric(shap_vals[, ncol(shap_vals)])
  if (!is.null(baseline) &&
    (length(baseline) == 1L || length(baseline) == n_rows)) {
    bias <- bias + as.numeric(baseline)
  }

  list(
    shap = shap_vals[, -ncol(shap_vals), drop = FALSE],
    bias = bias
  )
}

#' Extract Model Explanations (SHAP)
#'
#' Extracts exact tree-path-dependent Shapley Additive Explanations (TreeSHAP)
#' on the additive log-intensity (link) scale from a fitted `mlpp_model`,
#' including the model baseline offset so that `rowSums(shap) + bias` equals
#' `predict(object, res$data, type = "link")`.
#'
#' Note that TreeSHAP computes tree-path conditional expectations over the
#' fitted splits rather than interventional expectations against an external
#' background sample; the `background` argument selects which subset of `data`
#' observations (`"quadrature"`, `"events"`, or `"all"`) is evaluated and returned.
#'
#' @param object A fitted `mlpp_model` object.
#' @param data An `mlpp_data` object whose observations are to be explained.
#' @param background Character string specifying which subset of `data` rows
#'   to evaluate: `"quadrature"` (default, dummy points `y == 0`), `"events"`
#'   (observed event points `y == 1`), or `"all"` (all rows).
#' @param ... Additional arguments (currently unused).
#'
#' @details
#' TreeSHAP decomposes each observation's predicted log-intensity \eqn{\hat{\eta}(u)}
#' into an intercept term `bias` plus exact feature attributions \eqn{\phi_j(u)}
#' satisfying
#' \deqn{\hat{\eta}(u) = \mathrm{bias}(u) + \sum_{j=1}^{p} \phi_j(u).}
#' Subsetting via `background` selects which rows of `data` are evaluated without
#' altering the tree-path conditioning embedded in the fitted trees.
#'
#' @return A list of class `"mlpp_explanation"` containing:
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
#' @seealso [fit_gbpp()], [print.mlpp_explanation()], [plot.mlpp_explanation()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'     reticulate::py_available() &&
#'     reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   exp_obj <- explain_mlpp(fit, dat, background = "events")
#'   print(exp_obj)
#' }
#' @export
explain_mlpp <- function(
  object,
  data,
  background = c("quadrature", "events", "all"),
  ...
) {
  if (!inherits(object, "mlpp_model")) {
    stop("`object` must be a fitted `mlpp_model`.", call. = FALSE)
  }
  if (!inherits(data, "mlpp_data")) {
    stop("`data` must be an `mlpp_data` object.", call. = FALSE)
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

  res <- .get_shap(object, X_bg, ...)

  if (!is.null(baseline) && length(baseline) == length(y) && length(idx) != length(y)) {
    if (length(res$bias) == length(idx)) {
      res$bias <- res$bias + as.numeric(baseline)[idx]
    }
  }

  res$data <- X_bg

  class(res) <- c("mlpp_explanation", class(res))
  res
}

#' Print TreeSHAP Explanation Summary
#'
#' Prints evaluation sample count, feature count, mean baseline bias, and global
#' feature importance ranked by mean absolute SHAP value on the log-intensity scale.
#'
#' @param x An `mlpp_explanation` object produced by [explain_mlpp()].
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
#' @seealso [explain_mlpp()], [plot.mlpp_explanation()]
#'
#' @examples
#' exp_demo <- structure(
#'   list(
#'     shap = matrix(c(0.2, -0.1, 0.05, 0.3), ncol = 2, dimnames = list(NULL, c("x", "y"))),
#'     bias = c(-1.0, -1.0),
#'     data = data.frame(x = c(0.2, 0.8), y = c(0.4, 0.7))
#'   ),
#'   class = "mlpp_explanation"
#' )
#' print(exp_demo)
#' @keywords internal
#' @export
print.mlpp_explanation <- function(x, ...) {
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
#' @param x An `mlpp_explanation` object.
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
#' boxplots) against their corresponding SHAP contributions on the log-intensity
#' scale, overlaying a red `stats::lowess()` smoothing curve when a continuous
#' feature has more than 5 unique values.
#'
#' @return Invisibly returns `x`.
#'
#' @references
#' Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B.,
#' Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020).
#' From local explanations to global understanding with explainable AI for trees.
#' *Nature Machine Intelligence*, 2(1), 56--67.
#' \doi{10.1038/s42256-019-0138-9}
#'
#' @seealso [explain_mlpp()], [print.mlpp_explanation()]
#'
#' @examples
#' exp_demo <- structure(
#'   list(
#'     shap = matrix(c(0.2, -0.1, 0.05, 0.3), ncol = 2, dimnames = list(NULL, c("x", "y"))),
#'     bias = c(-1.0, -1.0),
#'     data = data.frame(x = c(0.2, 0.8), y = c(0.4, 0.7))
#'   ),
#'   class = "mlpp_explanation"
#' )
#' plot(exp_demo, type = "importance")
#' plot(exp_demo, type = "dependence", feature = "x")
#' @export
plot.mlpp_explanation <- function(x, type = c("importance", "dependence"), feature = NULL, ...) {
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
