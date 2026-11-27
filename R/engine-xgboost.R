# ============================================================
# mlpp - XGBoost Backend Engine
# ============================================================

#' Construct an XGBoost DMatrix
#'
#' @keywords internal
#' @noRd
.xgb_dmatrix <- function(
  X,
  y = NULL,
  base_margin = NULL,
  factor_levels = NULL,
  feature_names = NULL
) {
  xgb <- reticulate::import("xgboost", delay_load = FALSE)

  X_enc <- .encode_X(
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

#' Internal XGBoost engine fitting function
#'
#' @keywords internal
#' @noRd
.fit_xgb <- function(
  data,
  objective,
  params,
  n_trees,
  baseline = NULL,
  ...
) {
  .check_data(data, objective = objective)
  if (is.null(ncol(data$X)) || ncol(data$X) == 0L) {
    stop(
      "`data$X` must contain at least one covariate column to fit a tree model.",
      call. = FALSE
    )
  }
  .init_python()

  xgb <- reticulate::import("xgboost", delay_load = FALSE)

  if (!is.null(params$objective)) {
    warning(
      "Ignoring user-supplied `params$objective` (`",
      params$objective,
      "`); `mlpp` supplies a custom quadrature-weighted objective.",
      call. = FALSE
    )
    params$objective <- NULL
  }

  if (is.null(params$base_score)) {
    params$base_score <- 0.0
  }

  feature_names <- colnames(data$X)
  factor_levels <- .factor_levels(data$X)
  bl <- .init_baseline(data, objective, baseline)
  base_margin_vec <- if (bl$has_nonzero) bl$vec else NULL

  dtrain <- .xgb_dmatrix(
    X = data$X,
    y = data$y,
    base_margin = base_margin_vec,
    factor_levels = factor_levels,
    feature_names = feature_names
  )

  py_objective <- .make_obj(
    engine = "xgboost",
    objective = objective,
    v = data$quadrature_weight,
    delta = data$dummy_intensity,
    w = data$statistical_weight,
    A = data$interaction_integral
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
  .wrap_model(
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
