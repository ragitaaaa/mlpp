# ============================================================
# GBPP - Unified Prediction & Print Methods
# ============================================================

#' Predict from a Fitted GBPP Model
#'
#' Evaluates the fitted gradient boosted point process model on new covariate data,
#' returning the estimated log-intensity (link), spatial intensity, raw boosting tree margin,
#' or continuous spatial pixel images on planar or linear network domains.
#'
#' @param object A fitted model of class `gbpp_model`.
#' @param newdata Predictor covariates. Can be:
#'   - A `data.frame` or numeric `matrix` of covariate values evaluated at
#'     specific locations (for `type` in `"link"`, `"intensity"`, `"raw"`).
#'   - A named list of `spatstat.geom::im` objects (or an `owin` window) for
#'     2D continuous planar evaluation (for `type = "im"`).
#'   - A named list of `spatstat.linnet::linim` or `linfun` objects (or a
#'     `linnet` network) for linear network evaluation (for `type = "linim"`).
#' @param type Type of prediction:
#'   - `"link"` (default): The complete log-intensity linear predictor
#'     \eqn{\hat{\eta}(u) = \eta_0 + \sum_{k=1}^{K} f_k(z(u))}.
#'   - `"intensity"`: The spatial intensity \eqn{\hat{\lambda}(u) = \exp(\hat{\eta}(u))}.
#'   - `"raw"`: The raw ensemble contribution
#'     \eqn{\hat{F}(z(u)) = \sum_{k=1}^{K} f_k(z(u))} without baseline offset
#'     \eqn{\eta_0}.
#'   - `"im"`: A continuous 2D pixel image (`spatstat.geom::im`) of spatial
#'     intensity across a planar window.
#'   - `"linim"`: A continuous linear network pixel image
#'     (`spatstat.linnet::linim`) of spatial intensity across a network.
#' @param ... Additional arguments passed to the underlying engine prediction method
#'   (e.g., `eps` or `dimyx` when `type = "im"` or `type = "linim"`).
#'
#' @details
#' Given fitted tree predictions \eqn{\hat{F}(z(u)) = \sum_{k=1}^{K} f_k(z(u))}
#' and baseline offset \eqn{\eta_0}, `predict.gbpp_model()` returns:
#' - `"link"`: \eqn{\hat{\eta}(u) = \eta_0 + \hat{F}(z(u))};
#' - `"intensity"`: \eqn{\hat{\lambda}(u) = \exp(\hat{\eta}(u))}, clamped safely on the
#'   link scale to avoid floating-point overflow;
#' - `"raw"`: \eqn{\hat{F}(z(u))} without the baseline offset;
#' - `"im"` / `"linim"`: spatially mapped predictions on planar pixel grids or
#'   linear network pixel grids.
#'
#' Categorical columns are aligned automatically against the factor levels recorded
#' at training time (`attr(object, "factor_levels")`). Engine-specific methods
#' `predict.gbpp_xgboost()`, `predict.gbpp_lightgbm()`, and `predict.gbpp_catboost()`
#' delegate to `predict.gbpp_model()`.
#'
#' @return A numeric vector of predictions (for `type` in `"link"`, `"intensity"`, `"raw"`),
#'   an object of class `spatstat.geom::im` (for `type = "im"`), or an object of class
#'   `spatstat.linnet::linim` (for `type = "linim"`).
#'
#' @references
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' Lu, C., Guan, Y., van Lieshout, M.-C., & Xu, G. (2025). XGBoostPP: Tree-based
#' estimation of point process intensity functions. *Journal of Computational and
#' Graphical Statistics*, 35(1), 229--240. \doi{10.1080/10618600.2025.2520582}
#'
#' Choiruddin, A., Tasbihi, N. L., Abbad, M. A., Syukron, M., Widhianingsih,
#' T. D. A., & Mateu, J. (2026). XGBoost algorithm for a spatial point process
#' intensity estimation based on logistic loss. *Computational Journal of
#' Mathematical and Statistical Sciences*, 5(2), 472--496.
#' \doi{10.21608/cjmss.2026.452469.1353}
#'
#' @seealso [fit_gbpp()], [plot.gbpp_model()], [explain_gbpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_gbpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   head(predict(fit, newdata = dat$X, type = "intensity"))
#'   lam_im <- predict(fit, newdata = list(x = cov_x), type = "im", dimyx = c(16, 16))
#' }
#' @export
predict.gbpp_model <- function(
  object,
  newdata,
  type = c("link", "intensity", "raw", "im", "linim"),
  ...
) {
  type <- match.arg(type)

  if (type == "im") {
    return(.gbpp_predict_im(object, newdata, ...))
  }
  if (type == "linim") {
    return(.gbpp_predict_linim(object, newdata, ...))
  }

  if (missing(newdata) || is.null(newdata)) {
    stop("`newdata` must be supplied.", call. = FALSE)
  }

  if (!is.data.frame(newdata) && !is.matrix(newdata)) {
    stop("`newdata` must be a numeric matrix or data frame.", call. = FALSE)
  }

  if (inherits(object, "python.builtin.object")) {
    object <- .gbpp_rehydrate_model(object)
    raw_model <- object
    baseline <- attr(object, "baseline")
    baseline <- if (!is.null(baseline)) as.numeric(baseline) else 0.0
    engine <- attr(object, "engine")
    feature_names <- attr(object, "feature_names")
    factor_levels <- attr(object, "factor_levels")
    if (is.null(engine)) {
      engine <- if (inherits(object, "gbpp_xgboost")) {
        "xgboost"
      } else if (inherits(object, "gbpp_lightgbm")) {
        "lightgbm"
      } else if (inherits(object, "gbpp_catboost")) {
        "catboost"
      } else {
        stop("Unknown GBPP model engine.", call. = FALSE)
      }
    }
  } else if (is.list(object) && !is.null(object$raw_model)) {
    raw_model <- .gbpp_rehydrate_model(object$raw_model)
    baseline <- if (!is.null(object$baseline)) as.numeric(object$baseline) else 0.0
    engine <- object$engine
    feature_names <- object$feature_names
    factor_levels <- object$factor_levels
  } else {
    stop("Invalid `gbpp_model` object.", call. = FALSE)
  }

  # Standardize column schema, column ordering, and factor levels
  encoded <- .gbpp_encode_features(
    X = newdata,
    factor_levels = factor_levels,
    feature_names = feature_names,
    engine = engine,
    unseen = "error"
  )

  # Engine-specific raw prediction
  raw_pred <- switch(engine,
    xgboost = {
      configure_gbpp_python()
      mat <- as.matrix(encoded)
      if (!is.numeric(mat) || any(!is.finite(mat))) {
        stop("`newdata` must contain finite numeric values.", call. = FALSE)
      }

      xgb <- reticulate::import("xgboost", delay_load = FALSE)
      dmat_cols <- colnames(mat)
      bm_arr <- reticulate::np_array(rep(0.0, nrow(mat)), dtype = "float64")
      dtest <- if (!is.null(dmat_cols)) {
        xgb$DMatrix(
          data = mat,
          base_margin = bm_arr,
          feature_names = reticulate::r_to_py(as.list(dmat_cols))
        )
      } else {
        xgb$DMatrix(
          data = mat,
          base_margin = bm_arr
        )
      }

      dots <- list(...)
      pred <- if ("output_margin" %in% names(dots)) {
        raw_model$predict(dtest, ...)
      } else {
        raw_model$predict(dtest, output_margin = TRUE, ...)
      }
      as.numeric(reticulate::py_to_r(pred))
    },
    lightgbm = {
      configure_gbpp_python()
      mat <- as.matrix(encoded)
      if (!is.numeric(mat) || any(!is.finite(mat))) {
        stop("`newdata` must contain finite numeric values.", call. = FALSE)
      }

      pred <- raw_model$predict(data = mat, raw_score = TRUE, ...)
      as.numeric(reticulate::py_to_r(pred))
    },
    catboost = {
      configure_gbpp_python()
      cat_cols <- which(vapply(encoded, function(z) is.factor(z) || is.character(z), logical(1)))
      num_cols <- which(vapply(encoded, is.numeric, logical(1)))

      if (length(num_cols) > 0L) {
        num_vals <- encoded[, num_cols, drop = FALSE]
        if (any(!is.finite(as.matrix(num_vals)))) {
          stop("Numeric columns in `newdata` must contain only finite values.", call. = FALSE)
        }
      }
      if (length(cat_cols) > 0L) {
        for (j in cat_cols) {
          if (anyNA(encoded[[j]])) {
            stop("Categorical columns in `newdata` must not contain missing values.", call. = FALSE)
          }
        }
      }

      pred <- raw_model$predict(data = encoded, prediction_type = "RawFormulaVal", ...)
      as.numeric(reticulate::py_to_r(pred))
    }
  )

  if (type != "raw" && length(baseline) > 1L && length(baseline) != length(raw_pred)) {
    stop(
      "Model was fitted with a length-", length(baseline),
      " observation-specific `baseline` vector, which cannot be recycled for ",
      length(raw_pred), " prediction row(s).",
      call. = FALSE
    )
  }

  # Apply baseline and link transformation
  switch(type,
    raw = raw_pred,
    link = baseline + raw_pred,
    intensity = exp(pmin(pmax(baseline + raw_pred, -50), 50))
  )
}

#' @rdname predict.gbpp_model
#' @export
predict.gbpp_xgboost <- function(object, newdata, ...) {
  predict.gbpp_model(object = object, newdata = newdata, ...)
}

#' @rdname predict.gbpp_model
#' @export
predict.gbpp_lightgbm <- function(object, newdata, ...) {
  predict.gbpp_model(object = object, newdata = newdata, ...)
}

#' @rdname predict.gbpp_model
#' @export
predict.gbpp_catboost <- function(object, newdata, ...) {
  predict.gbpp_model(object = object, newdata = newdata, ...)
}

#' Print a GBPP fitted model
#'
#' Prints a concise summary of a fitted `gbpp_model` object, including the
#' boosting backend engine, objective, tree count, baseline offset, and feature names.
#'
#' @param x A `gbpp_model` object.
#' @param ... Additional arguments (currently unused).
#'
#' @details
#' Displays both scalar baselines (\eqn{\eta_0} and \eqn{\lambda_0 = \exp(\eta_0)})
#' and summary statistics when observation-specific baseline vectors are used.
#'
#' @return Invisibly returns `x`.
#'
#' @seealso [fit_gbpp()], [predict.gbpp_model()], [explain_gbpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_gbpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   print(fit)
#' }
#' @export
print.gbpp_model <- function(x, ...) {
  bl <- attr(x, "baseline")
  cat("Gradient Boosted Point Process Model\n")
  cat("------------------------------------\n")
  cat("Engine          :", attr(x, "engine"), "\n")
  cat("Objective       :", attr(x, "objective"), "\n")
  cat("Number of trees :", attr(x, "n_trees"), "\n")
  if (!is.null(bl) && length(bl) > 1L) {
    cat(
      "Baseline (eta0) : vector (mean =", round(mean(bl), 4),
      sprintf(", mean lambda0 = %.4f)\n", mean(exp(pmin(pmax(bl, -50), 50))))
    )
  } else if (!is.null(bl) && length(bl) == 1L) {
    cat("Baseline (eta0) :", round(bl, 4), sprintf("(lambda0 = %.4f)\n", exp(bl)))
  } else {
    cat("Baseline (eta0) : 0 (lambda0 = 1.0000)\n")
  }
  cat("Features (p)    :", length(attr(x, "feature_names")), "\n")
  if (length(attr(x, "feature_names")) > 0L) {
    cat("Feature names   :", paste(attr(x, "feature_names"), collapse = ", "), "\n")
  }
  invisible(x)
}
