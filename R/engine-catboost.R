# ============================================================
# mlpp - CatBoost Backend Engine
# ============================================================

#' Internal CatBoost engine fitting function
#'
#' @keywords internal
#' @noRd
.fit_cb <- function(
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

  catboost <- reticulate::import("catboost", delay_load = FALSE)
  feature_names <- colnames(data$X)
  factor_levels <- .factor_levels(data$X)

  cat_features_idx <- NULL

  if (is.matrix(data$X)) {
    if (!is.numeric(data$X)) {
      stop("`data$X` matrix must be numeric.", call. = FALSE)
    }
    if (any(!is.finite(data$X))) {
      stop("`data$X` must contain only finite values.", call. = FALSE)
    }
  } else if (is.data.frame(data$X)) {
    categorical_cols <- which(
      vapply(data$X, function(z) is.factor(z) || is.character(z), logical(1))
    )
    numeric_cols <- which(vapply(data$X, is.numeric, logical(1)))

    if (length(numeric_cols) > 0L) {
      numeric_values <- data$X[, numeric_cols, drop = FALSE]
      if (any(!is.finite(as.matrix(numeric_values)))) {
        stop("Numeric columns in `data$X` must contain only finite values.", call. = FALSE)
      }
    }

    if (length(categorical_cols) > 0L) {
      for (j in categorical_cols) {
        if (anyNA(data$X[[j]])) {
          stop(
            "Categorical columns in `data$X` must not contain missing values.",
            call. = FALSE
          )
        }
      }
      cat_features_idx <- as.integer(unname(categorical_cols - 1L))
      if (!("cat_features" %in% names(params))) {
        params$cat_features <- as.list(cat_features_idx)
      }
    }
  }

  X <- .encode_X(
    data$X,
    factor_levels = factor_levels,
    feature_names = feature_names,
    engine = "catboost"
  )

  bl <- .init_baseline(data, objective, baseline)
  base_vec <- if (!is.null(bl$vec)) bl$vec else rep(0.0, nrow(X))

  if (is.null(params$allow_writing_files)) {
    params$allow_writing_files <- FALSE
  }
  if (is.null(params$one_hot_max_size)) {
    params$one_hot_max_size <- 0L
  }

  is_iterative <- objective %in% c("iterative_poisson", "iterative_logistic")

  if (is_iterative) {
    if (!is.null(params$boosting_type) &&
      identical(as.character(params$boosting_type), "Ordered")) {
      stop(
        "CatBoost `boosting_type = 'Ordered'` is incompatible with stateful ",
        "`iterative_*` objectives because dynamic interaction weights depend on ",
        "per-observation `interaction_integral`. Use `boosting_type = 'Plain'` or ",
        "use `poisson`/`logistic` with `Ordered` boosting.",
        call. = FALSE
      )
    }
    params$boosting_type <- "Plain"
    params$has_time <- TRUE
    params$thread_count <- 1L
    if (!is.null(params$subsample) && as.numeric(params$subsample) < 1.0) {
      warning(
        "CatBoost `iterative_*` objectives require full-batch evaluation; ",
        "overriding `subsample` to 1.0.",
        call. = FALSE
      )
      params$subsample <- 1.0
      params$bootstrap_type <- NULL
    }

    py_objective <- .make_obj(
      engine = "catboost",
      objective = objective,
      v = data$quadrature_weight,
      delta = data$dummy_intensity,
      A = data$interaction_integral
    )

    train_pool <- catboost$Pool(
      data = X,
      label = as.numeric(data$y),
      baseline = if (bl$has_nonzero) base_vec else NULL,
      cat_features = params$cat_features
    )
  } else {
    pool_baseline <- switch(objective,
      poisson = base_vec + log(pmax(data$quadrature_weight, 1e-300)),
      weighted_poisson = base_vec + log(pmax(data$quadrature_weight, 1e-300)),
      logistic = base_vec - log(data$dummy_intensity),
      weighted_logistic = base_vec - log(data$dummy_intensity)
    )

    pool_weight <- if (objective %in% c("weighted_poisson", "weighted_logistic")) {
      as.numeric(data$statistical_weight)
    } else {
      NULL
    }

    py_objective <- .make_obj(
      engine = "catboost",
      objective = objective
    )

    train_pool <- catboost$Pool(
      data = X,
      label = as.numeric(data$y),
      weight = pool_weight,
      baseline = as.numeric(pool_baseline),
      cat_features = params$cat_features
    )
  }

  if (!is.null(params$loss_function)) {
    warning(
      "Ignoring user-supplied `params$loss_function`; `mlpp` supplies a custom quadrature-weighted objective.",
      call. = FALSE
    )
  }

  params$loss_function <- py_objective
  params$iterations <- as.integer(n_trees)

  raw_model <- do.call(catboost$CatBoostRegressor, params)

  extra_args <- list(...)
  extra_args$X <- NULL
  extra_args$y <- NULL
  fit_args <- c(list(X = train_pool), extra_args)

  do.call(raw_model$fit, fit_args)

  params_meta <- params
  params_meta$loss_function <- objective

  .wrap_model(
    raw_model = raw_model,
    engine = "catboost",
    objective = objective,
    baseline_attr = bl$attr_val,
    feature_names = feature_names,
    factor_levels = factor_levels,
    params = params_meta,
    n_trees = n_trees,
    data = data,
    extra_attrs = list(cat_features = cat_features_idx)
  )
}
