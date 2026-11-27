# ============================================================
# mlpp - Unified Prediction, Spatial Mapping & Print Methods
# ============================================================

#' Predict from a Fitted mlpp Model
#'
#' Evaluates a fitted point process model on new covariate data, returning the
#' estimated log-intensity (link), spatial intensity, raw model margin, or
#' continuous spatial pixel images on planar (`im`) or linear network (`linim`)
#' domains.
#'
#' @param object A fitted model of class `mlpp_model`.
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
#'   - `"im"`: A continuous 2D pixel image (`spatstat.geom::im`) across a planar
#'     window (supports `scale = c("intensity", "link")`, `eps`, and `dimyx` via `...`).
#'   - `"linim"`: A continuous linear network pixel image
#'     (`spatstat.linnet::linim`) across a network (supports
#'     `scale = c("intensity", "link")` and `eps` via `...`).
#' @param ... Additional arguments passed to spatial rasterizers (`eps`, `dimyx`,
#'   `scale = c("intensity", "link")` when `type` is `"im"` or `"linim"`) or to
#'   the underlying engine `predict()` method.
#'
#' @details
#' Given fitted tree predictions \eqn{\hat{F}(z(u)) = \sum_{k=1}^{K} f_k(z(u))}
#' and baseline offset \eqn{\eta_0}, `predict.mlpp_model()` returns:
#' - `"link"`: \eqn{\hat{\eta}(u) = \eta_0 + \hat{F}(z(u))};
#' - `"intensity"`: \eqn{\hat{\lambda}(u) = \exp(\hat{\eta}(u))}, clamped to
#'   `[-50, 50]` on the link scale to avoid floating-point overflow;
#' - `"raw"`: \eqn{\hat{F}(z(u))} without the baseline offset;
#' - `"im"` / `"linim"`: spatially mapped predictions on planar pixel grids or
#'   linear network pixel grids (`scale = "intensity"` by default, or `"link"`).
#'
#' Categorical columns are aligned automatically against the factor levels recorded
#' at training time (`attr(object, "factor_levels")`), raising an error on unseen
#' levels. Models fitted with an observation-specific vector `baseline` require
#' `nrow(newdata) == length(baseline)` for `type` in `"link"`, `"intensity"`,
#' `"im"`, or `"linim"` (use `type = "raw"` for arbitrary subsets).
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
#' @seealso [fit_gbpp()], [plot.mlpp_model()], [eval_mlpp()], [explain_mlpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   head(predict(fit, newdata = dat$X, type = "intensity"))
#'   lam_im <- predict(fit, newdata = list(x = cov_x), type = "im", dimyx = c(16, 16))
#' }
#' @export
predict.mlpp_model <- function(
  object,
  newdata,
  type = c("link", "intensity", "raw", "im", "linim"),
  ...
) {
  type <- match.arg(type)

  if (type == "im") {
    return(.pred_im(object, newdata, ...))
  }
  if (type == "linim") {
    return(.pred_linim(object, newdata, ...))
  }

  if (!inherits(object, "mlpp_model")) {
    stop("`object` must be an `mlpp_model` object.", call. = FALSE)
  }

  if (missing(newdata) || is.null(newdata)) {
    stop("`newdata` must be supplied.", call. = FALSE)
  }

  if (!is.data.frame(newdata) && !is.matrix(newdata)) {
    stop("`newdata` must be a numeric matrix or data frame.", call. = FALSE)
  }

  object <- .unpack_model(object)
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
      stop("Unknown mlpp model engine.", call. = FALSE)
    }
  }

  encoded <- .encode_X(
    X = newdata,
    factor_levels = factor_levels,
    feature_names = feature_names,
    engine = engine,
    unseen = "error"
  )

  raw_pred <- switch(engine,
    xgboost = {
      .init_python()
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
      .init_python()
      mat <- as.matrix(encoded)
      if (!is.numeric(mat) || any(!is.finite(mat))) {
        stop("`newdata` must contain finite numeric values.", call. = FALSE)
      }

      pred <- raw_model$predict(data = mat, raw_score = TRUE, ...)
      as.numeric(reticulate::py_to_r(pred))
    },
    catboost = {
      .init_python()
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

  switch(type,
    raw = raw_pred,
    link = baseline + raw_pred,
    intensity = exp(pmin(pmax(baseline + raw_pred, -50), 50))
  )
}

#' Print a fitted mlpp model
#'
#' Prints a concise summary of a fitted `mlpp_model` object, including the
#' engine, objective, tree count, baseline offset, and feature names.
#'
#' @param x An `mlpp_model` object.
#' @param ... Additional arguments (currently unused).
#'
#' @details
#' Displays both scalar baselines (\eqn{\eta_0} and \eqn{\lambda_0 = \exp(\eta_0)})
#' and summary statistics when observation-specific baseline vectors are used.
#'
#' @return Invisibly returns `x`.
#'
#' @seealso [fit_gbpp()], [predict.mlpp_model()], [explain_mlpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   print(fit)
#' }
#' @keywords internal
#' @export
print.mlpp_model <- function(x, ...) {
  bl <- attr(x, "baseline")
  cat("MLPP Fitted Point Process Model\n")
  cat("-------------------------------\n")
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

#' Map mlpp predictions to a planar spatstat pixel image (`im`)
#'
#' @keywords internal
#' @noRd
.pred_im <- function(
  object,
  newdata,
  eps = NULL,
  dimyx = NULL,
  scale = c("intensity", "link"),
  ...
) {
  scale <- match.arg(scale)

  if (missing(newdata) || is.null(newdata)) {
    stop(
      "`newdata` must be a list of `spatstat.geom::im` objects or an `owin` window.",
      call. = FALSE
    )
  }

  if (inherits(newdata, "owin")) {
    window <- newdata
    covariates <- list()
  } else if (is.list(newdata) &&
    length(newdata) > 0 &&
    all(vapply(newdata, inherits, logical(1), "im"))) {
    window <- spatstat.geom::Window(newdata[[1]])
    covariates <- newdata
    disjoint <- vapply(
      covariates,
      function(z) {
        inter <- spatstat.geom::intersect.owin(window, spatstat.geom::Window(z), fatal = FALSE)
        is.null(inter) || spatstat.geom::is.empty(inter) || spatstat.geom::area(inter) <= 0
      },
      logical(1)
    )
    if (any(disjoint)) {
      stop("All `im` covariates in `newdata` must overlap a common spatial window.", call. = FALSE)
    }
  } else {
    stop(
      "`newdata` for type = 'im' must be a list of `im` objects or an `owin` window.",
      call. = FALSE
    )
  }

  feature_names <- attr(object, "feature_names")

  if (length(feature_names) > 0) {
    required_covs <- setdiff(feature_names, c("x", "y"))
    if (length(required_covs) > 0 && !all(required_covs %in% names(covariates))) {
      missing_covs <- setdiff(required_covs, names(covariates))
      stop(
        sprintf(
          "Missing required covariates in `newdata`: %s",
          paste(missing_covs, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }

  mask <- spatstat.geom::as.mask(window, eps = eps, dimyx = dimyx)
  coords <- spatstat.geom::raster.xy(mask, drop = FALSE)
  n_pts <- length(coords$x)

  df <- if (length(feature_names) > 0L) {
    cols_list <- vector("list", length(feature_names))
    names(cols_list) <- feature_names
    for (f in feature_names) {
      if (f %in% names(covariates)) {
        v_extracted <- covariates[[f]][list(x = coords$x, y = coords$y)]
        if (is.factor(v_extracted) || is.character(v_extracted)) {
          cols_list[[f]] <- as.character(v_extracted)
        } else {
          cols_list[[f]] <- as.numeric(v_extracted)
        }
      } else if (f == "x") {
        cols_list[[f]] <- as.numeric(coords$x)
      } else if (f == "y") {
        cols_list[[f]] <- as.numeric(coords$y)
      }
    }
    as.data.frame(cols_list, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    data.frame(x = as.numeric(coords$x), y = as.numeric(coords$y))[seq_len(n_pts), 0, drop = FALSE]
  }

  inside <- as.vector(mask$m)
  valid_covariates <- stats::complete.cases(df)
  valid <- inside & valid_covariates

  out_mat <- matrix(NA_real_, nrow = mask$dim[1], ncol = mask$dim[2])

  if (any(valid)) {
    df_valid <- df[valid, feature_names, drop = FALSE]
    preds <- predict.mlpp_model(object, newdata = df_valid, type = scale, ...)
    out_mat[valid] <- preds
  }

  spatstat.geom::im(out_mat, xcol = mask$xcol, yrow = mask$yrow)
}

#' Map mlpp predictions to a linear-network spatstat pixel image (`linim`)
#'
#' @keywords internal
#' @noRd
.pred_linim <- function(
  object,
  newdata,
  eps = NULL,
  dimyx = NULL,
  scale = c("intensity", "link"),
  ...
) {
  scale <- match.arg(scale)

  if (missing(newdata) || is.null(newdata)) {
    stop(
      "`newdata` must be a list of `spatstat.linnet::linim` objects or a ",
      "`linnet` object.",
      call. = FALSE
    )
  }

  if (inherits(newdata, "linnet")) {
    network <- newdata
    covariates <- list()
  } else if (is.list(newdata) && length(newdata) > 0 &&
    all(vapply(newdata, function(x) inherits(x, c("linim", "linfun")), logical(1)))) {
    network <- spatstat.linnet::as.linnet(newdata[[1]])
    .check_linnet(
      newdata,
      network,
      "All network covariates in `newdata` must be defined on the same linear network."
    )
    covariates <- newdata
  } else {
    stop(
      "`newdata` for type = 'linim' must be a list of `linim` objects, ",
      "`linfun` objects, or a `linnet` object.",
      call. = FALSE
    )
  }

  feature_names <- attr(object, "feature_names")

  if (length(feature_names) > 0) {
    required_covs <- setdiff(feature_names, c("x", "y"))
    if (length(required_covs) > 0 && !all(required_covs %in% names(covariates))) {
      missing_covs <- setdiff(required_covs, names(covariates))
      stop(
        sprintf(
          "Missing required covariates in `newdata`: %s",
          paste(missing_covs, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }

  f_predict <- function(x, y, seg, tp) {
    n_pts <- length(x)
    x_num <- as.numeric(x)
    y_num <- as.numeric(y)

    needs_lpp <- any(vapply(
      intersect(feature_names, names(covariates)),
      function(nm) !inherits(covariates[[nm]], "linfun"),
      logical(1)
    ))

    p_lpp <- NULL
    if (needs_lpp) {
      p_lpp <- tryCatch(
        spatstat.linnet::lpp(
          data.frame(x = x_num, y = y_num, seg = as.integer(seg), tp = as.numeric(tp)),
          L = network,
          check = FALSE
        ),
        error = function(e) {
          spatstat.linnet::as.lpp(
            x = x_num, y = y_num, seg = as.integer(seg), tp = as.numeric(tp),
            L = network
          )
        }
      )
    }

    cols_list <- vector("list", length(feature_names))
    names(cols_list) <- feature_names

    for (f in feature_names) {
      if (f %in% names(covariates)) {
        cov_obj <- covariates[[f]]
        v_raw <- if (inherits(cov_obj, "linfun")) {
          cov_obj(x_num, y_num, seg, tp)
        } else {
          cov_obj[p_lpp]
        }
        if (length(v_raw) != n_pts) {
          stop("Network covariate evaluation length mismatch for `", f, "`.", call. = FALSE)
        }
        if (is.factor(v_raw) || is.character(v_raw)) {
          cols_list[[f]] <- as.character(v_raw)
        } else {
          cols_list[[f]] <- as.numeric(v_raw)
        }
      } else if (f == "x") {
        cols_list[[f]] <- x_num
      } else if (f == "y") {
        cols_list[[f]] <- y_num
      }
    }

    df <- as.data.frame(cols_list, stringsAsFactors = FALSE, check.names = FALSE)
    valid <- stats::complete.cases(df)
    out <- rep(NA_real_, n_pts)

    if (any(valid)) {
      df_valid <- df[valid, feature_names, drop = FALSE]
      out[valid] <- predict.mlpp_model(object, newdata = df_valid, type = scale, ...)
    }
    out
  }

  lf <- spatstat.linnet::linfun(f_predict, network)
  spatstat.linnet::as.linim(lf, eps = eps)
}
