# ============================================================
# GBPP - XGBoost Backend Engine
# ============================================================

#' Import the XGBoost Python module for GBPP
#'
#' @keywords internal
#' @noRd
.gbpp_xgboost_module <- function() {
  configure_gbpp_python()
  python_path <- get_gbpp_python()

  reticulate::import_from_path(
    module = "gbpp_xgboost",
    path = python_path
  )
}

#' Construct an XGBoost DMatrix for GBPP
#'
#' @keywords internal
#' @noRd
.gbpp_xgboost_dmatrix <- function(
  X,
  y = NULL,
  base_margin = NULL,
  factor_levels = NULL,
  feature_names = NULL
) {
  xgb <- reticulate::import("xgboost", delay_load = FALSE)

  X_enc <- .gbpp_encode_features(
    X,
    factor_levels = factor_levels,
    feature_names = feature_names,
    engine = "xgboost"
  )

  X_matrix <- as.matrix(X_enc)
  col_names <- colnames(X_matrix)

  labels <- if (!is.null(y)) as.numeric(y) else NULL

  if (!is.null(base_margin)) {
    base_margin <- as.numeric(base_margin)
    if (length(base_margin) != nrow(X_matrix)) {
      stop("`base_margin` must have the same length as observations.", call. = FALSE)
    }
    base_margin <- reticulate::np_array(base_margin, dtype = "float64")
  }

  if (is.null(col_names)) {
    xgb$DMatrix(
      data = X_matrix,
      label = labels,
      base_margin = base_margin
    )
  } else {
    xgb$DMatrix(
      data = X_matrix,
      label = labels,
      base_margin = base_margin,
      feature_names = as.list(col_names)
    )
  }
}

# ============================================================
# Internal Engine Fitting Function
# ============================================================

#' Internal XGBoost engine fitting function
#'
#' @keywords internal
#' @noRd
.gbpp_fit_xgboost <- function(
  data,
  objective,
  params,
  n_trees,
  baseline = NULL,
  ...
) {
  validate_gbpp_data(data, objective = objective)
  if (is.null(ncol(data$X)) || ncol(data$X) == 0L) {
    stop(
      "`data$X` must contain at least one covariate column to fit a tree model.",
      call. = FALSE
    )
  }
  configure_gbpp_python()

  xgb <- reticulate::import("xgboost", delay_load = FALSE)

  if (!is.null(params$objective)) {
    warning(
      "Ignoring user-supplied `params$objective` (`",
      params$objective,
      "`); `gbpp` supplies a custom quadrature-weighted objective.",
      call. = FALSE
    )
    params$objective <- NULL
  }

  if (is.null(params$base_score)) {
    params$base_score <- 0.0
  }

  feature_names <- colnames(data$X)
  factor_levels <- .gbpp_extract_factor_levels(data$X)
  bl <- .gbpp_init_baseline(data, objective, baseline)
  base_margin_vec <- if (bl$has_nonzero) bl$vec else NULL

  dtrain <- .gbpp_xgboost_dmatrix(
    X = data$X,
    y = data$y,
    base_margin = base_margin_vec,
    factor_levels = factor_levels,
    feature_names = feature_names
  )

  py_objective <- .gbpp_make_objective(
    engine = "xgboost",
    objective = objective,
    quadrature_weight = data$quadrature_weight,
    dummy_intensity = data$dummy_intensity,
    statistical_weight = data$statistical_weight,
    interaction_integral = data$interaction_integral
  )

  extra_args <- list(...)
  extra_args$num_boost_round <- NULL
  extra_args$nrounds <- NULL

  train_args <- c(
    list(
      params = params,
      dtrain = dtrain,
      num_boost_round = as.integer(n_trees),
      obj = py_objective
    ),
    extra_args
  )

  raw_model <- do.call(xgb$train, train_args)
  .gbpp_finalize_model(
    raw_model = raw_model,
    engine = "xgboost",
    objective = objective,
    baseline_attr = bl$attr_val,
    feature_names = feature_names,
    factor_levels = factor_levels,
    params = params,
    n_trees = n_trees,
    data = data
  )
}

# ============================================================
# SHAP Extraction
# ============================================================

#' @export
.gbpp_shap.gbpp_xgboost <- function(object, newdata, ...) {
  object <- .gbpp_rehydrate_model(object)

  feat_names <- attr(object, "feature_names")
  n_rows <- nrow(newdata)

  dmat <- .gbpp_xgboost_dmatrix(
    X = newdata,
    base_margin = rep(0.0, n_rows),
    factor_levels = attr(object, "factor_levels"),
    feature_names = feat_names
  )
  shap_vals <- object$predict(dmat, pred_contribs = TRUE)

  .gbpp_finalize_shap(shap_vals, object, n_rows)
}
