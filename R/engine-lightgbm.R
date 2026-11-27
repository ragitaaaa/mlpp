# ============================================================
# mlpp - LightGBM Backend Engine
# ============================================================

#' Internal LightGBM engine fitting function
#'
#' @keywords internal
#' @noRd
.fit_lgb <- function(
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

  lgb <- reticulate::import("lightgbm", delay_load = FALSE)

  feature_names <- colnames(data$X)
  factor_levels <- .factor_levels(data$X)
  bl <- .init_baseline(data, objective, baseline)
  init_score <- if (bl$has_nonzero) bl$vec else NULL

  if (nrow(data$X) < 20L) {
    if (is.null(params$min_data_in_bin) || params$min_data_in_bin > 1L) {
      params$min_data_in_bin <- 1L
    }
    if (is.null(params$min_data_in_leaf) || params$min_data_in_leaf > 1L) {
      params$min_data_in_leaf <- 1L
    }
    if (is.null(params$min_sum_hessian_in_leaf) || params$min_sum_hessian_in_leaf > 1e-3) {
      params$min_sum_hessian_in_leaf <- 1e-3
    }
  }

  ds_params <- list()
  if (!is.null(params$min_data_in_bin)) {
    ds_params$min_data_in_bin <- as.integer(params$min_data_in_bin)
  }
  if (!is.null(params$min_data_in_leaf)) {
    ds_params$min_data_in_leaf <- as.integer(params$min_data_in_leaf)
  }

  X_data <- .encode_X(
    data$X,
    factor_levels = factor_levels,
    feature_names = feature_names,
    engine = "lightgbm"
  )

  lgb_feat_names <- if (!is.null(feature_names)) {
    reticulate::r_to_py(as.list(feature_names))
  } else {
    "auto"
  }

  dtrain <- lgb$Dataset(
    data = as.matrix(X_data),
    label = data$y,
    feature_name = lgb_feat_names,
    init_score = init_score,
    params = if (length(ds_params) > 0) ds_params else NULL
  )

  py_objective <- .make_obj(
    engine = "lightgbm",
    objective = objective,
    v = data$quadrature_weight,
    delta = data$dummy_intensity,
    w = data$statistical_weight,
    A = data$interaction_integral
  )

  if (!is.null(params$objective)) {
    warning(
      "Ignoring user-supplied `params$objective`; `mlpp` supplies a custom quadrature-weighted objective.",
      call. = FALSE
    )
  }

  params$objective <- py_objective

  extra_args <- list(...)
  extra_args$num_boost_round <- NULL
  extra_args$nrounds <- NULL

  train_args <- c(
    list(
      params = params,
      train_set = dtrain,
      num_boost_round = as.integer(n_trees)
    ),
    extra_args
  )

  raw_model <- do.call(lgb$train, train_args)
  params_meta <- params
  params_meta$objective <- objective

  .wrap_model(
    raw_model = raw_model,
    engine = "lightgbm",
    objective = objective,
    baseline_attr = bl$attr_val,
    feature_names = feature_names,
    factor_levels = factor_levels,
    params = params_meta,
    n_trees = n_trees,
    data = data
  )
}
