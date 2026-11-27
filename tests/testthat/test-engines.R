test_that(".norm_params standardizes parameters and cross-engine aliases", {
  xgb_res <- mlpp:::.norm_params(
    engine = "xgboost",
    n_trees = 50L,
    learning_rate = 0.03,
    max_depth = 4L,
    subsample = 0.8,
    colsample_bytree = 0.7,
    random_seed = 42L,
    verbose = FALSE,
    user_params = list(learning_rate = 0.09, gamma = 0.5)
  )
  expect_equal(xgb_res$n_trees, 50L)
  expect_equal(xgb_res$params$eta, 0.09)
  expect_null(xgb_res$params$learning_rate)
  expect_equal(xgb_res$params$gamma, 0.5)

  lgb_res <- mlpp:::.norm_params(
    engine = "lightgbm",
    n_trees = 75L,
    user_params = list(subsample = 0.8, colsample_bytree = 0.7, max_depth = 4L)
  )
  expect_equal(lgb_res$params$bagging_fraction, 0.8)
  expect_equal(lgb_res$params$bagging_freq, 1L)
  expect_equal(lgb_res$params$feature_fraction, 0.7)
  expect_equal(lgb_res$params$num_leaves, 16L)
  expect_null(lgb_res$params$subsample)

  cat_res <- mlpp:::.norm_params(
    engine = "catboost",
    n_trees = 100L,
    subsample = 0.8,
    user_params = list(max_depth = 4L, seed = 77L, colsample_bytree = 0.7)
  )
  expect_equal(cat_res$params$depth, 4L)
  expect_null(cat_res$params$max_depth)
  expect_equal(cat_res$params$random_seed, 77L)
  expect_null(cat_res$params$seed)
  expect_equal(cat_res$params$rsm, 0.7)
  expect_null(cat_res$params$colsample_bytree)
  expect_equal(cat_res$params$bootstrap_type, "Bernoulli")
  expect_equal(cat_res$params$one_hot_max_size, 0L)

  expect_error(mlpp:::.norm_params("xgboost", n_trees = -1L), "n_trees")
  expect_error(mlpp:::.norm_params("xgboost", learning_rate = -0.1), "learning_rate")
  expect_error(mlpp:::.norm_params("xgboost", max_depth = 0L), "max_depth")
  expect_error(mlpp:::.norm_params("xgboost", subsample = 1.5), "subsample")
  expect_error(mlpp:::.norm_params("xgboost", colsample_bytree = 0), "colsample_bytree")
})

test_that("XGBoost DMatrix preserves feature count and names", {
  skip_if_no_python("xgboost")

  d1 <- mlpp:::.xgb_dmatrix(X = data.frame(numeric = 1:5), y = c(0, 1, 0, 1, 0))
  expect_equal(d1$num_col(), 1L)
  expect_identical(reticulate::py_to_r(d1$feature_names), "numeric")

  d2 <- mlpp:::.xgb_dmatrix(X = data.frame(x1 = 1:5, x2 = 5:1), y = c(0, 1, 0, 1, 0))
  expect_equal(d2$num_col(), 2L)
  expect_identical(reticulate::py_to_r(d2$feature_names), c("x1", "x2"))
})

test_that("XGBoost, LightGBM, and CatBoost match R reference derivatives across all 6 objectives", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  data <- make_small_mlpp_data()
  data$statistical_weight <- c(1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.0)
  prediction <- seq(-0.6, 0.8, length.out = nrow(data$X))

  objectives <- c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )

  xgb <- reticulate::import("xgboost", delay_load = FALSE)
  dtrain_xgb <- xgb$DMatrix(data = as.matrix(data$X), label = as.numeric(data$y))

  lgb <- reticulate::import("lightgbm", delay_load = FALSE)
  dtrain_lgb <- lgb$Dataset(data = as.matrix(data$X), label = as.numeric(data$y))
  dtrain_lgb$construct()

  extract_cb <- function(py_obj, approx, targets, weights = NULL) {
    ders <- py_obj$calc_ders_range(
      reticulate::r_to_py(as.list(approx)),
      reticulate::r_to_py(as.list(as.numeric(targets))),
      if (is.null(weights)) NULL else reticulate::r_to_py(as.list(as.numeric(weights)))
    )
    list(
      gradient = -vapply(ders, function(pair) as.numeric(pair[[1]]), numeric(1)),
      hessian = -vapply(ders, function(pair) as.numeric(pair[[2]]), numeric(1))
    )
  }

  for (obj_name in objectives) {
    expected <- mlpp:::.ref_deriv(data = data, eta = prediction, objective = obj_name)

    py_xgb <- mlpp:::.make_obj(
      "xgboost", obj_name,
      v = data$quadrature_weight,
      delta = data$dummy_intensity,
      w = data$statistical_weight,
      A = data$interaction_integral
    )
    res_xgb <- py_xgb(reticulate::r_to_py(prediction), dtrain_xgb)
    expect_equal(as.numeric(reticulate::py_to_r(res_xgb[[1]])), expected$gradient, tolerance = 1e-12)
    expect_equal(as.numeric(reticulate::py_to_r(res_xgb[[2]])), expected$hessian, tolerance = 1e-12)

    py_lgb <- mlpp:::.make_obj(
      "lightgbm", obj_name,
      v = data$quadrature_weight,
      delta = data$dummy_intensity,
      w = data$statistical_weight,
      A = data$interaction_integral
    )
    res_lgb <- py_lgb(reticulate::r_to_py(prediction), dtrain_lgb)
    expect_equal(as.numeric(reticulate::py_to_r(res_lgb[[1]])), expected$gradient, tolerance = 1e-12)
    expect_equal(as.numeric(reticulate::py_to_r(res_lgb[[2]])), expected$hessian, tolerance = 1e-12)

    if (obj_name %in% c("iterative_poisson", "iterative_logistic")) {
      py_cb <- mlpp:::.make_obj(
        "catboost", obj_name,
        v = data$quadrature_weight,
        delta = data$dummy_intensity,
        w = data$statistical_weight,
        A = data$interaction_integral
      )
      got_cb <- extract_cb(py_cb, prediction, data$y)
      expect_equal(got_cb$gradient, expected$gradient, tolerance = 1e-12)
      expect_equal(got_cb$hessian, expected$hessian, tolerance = 1e-12)
    }
  }

  for (obj_name in c("poisson", "weighted_poisson", "logistic", "weighted_logistic")) {
    expected <- mlpp:::.ref_deriv(data = data, eta = prediction, objective = obj_name)
    py_cb_sl <- mlpp:::.make_obj("catboost", obj_name)
    approx_shifted <- if (grepl("poisson", obj_name)) {
      prediction + log(data$quadrature_weight)
    } else {
      prediction - log(data$dummy_intensity)
    }
    pool_w <- if (grepl("weighted", obj_name)) data$statistical_weight else NULL
    got_sl <- extract_cb(py_cb_sl, approx_shifted, data$y, weights = pool_w)
    expect_equal(got_sl$gradient, expected$gradient, tolerance = 1e-12)
    expect_equal(got_sl$hessian, expected$hessian, tolerance = 1e-12)
  }
})

test_that("fit_gbpp strips params$objective and predict enforces output_margin for xgboost", {
  skip_if_not_installed("spatstat.geom")
  skip_if_no_python("xgboost")

  set.seed(123)
  win <- spatstat.geom::owin(c(0, 10), c(0, 10))
  pts <- spatstat.geom::ppp(x = runif(200, 0, 10), y = runif(200, 0, 10), window = win)
  X <- data.frame(x1 = rnorm(200), f1 = factor(sample(c("a", "b", "c"), 200, replace = TRUE)))
  y <- c(rep(1L, 40L), rep(0L, 160L))
  w <- rep(100 / 160, 200)
  dat <- mlpp_data(X = X, y = y, locations = pts, quadrature_weight = w, window = win, type = "planar")

  fit_clean <- fit_gbpp(dat, engine = "xgboost", n_trees = 15L, learning_rate = 0.1, max_depth = 3L, random_seed = 42L)
  expect_warning(
    fit_guarded <- fit_gbpp(
      dat,
      engine = "xgboost",
      n_trees = 15L,
      learning_rate = 0.1,
      max_depth = 3L,
      random_seed = 42L,
      params = list(objective = "count:poisson")
    ),
    "Ignoring user-supplied `params\\$objective`"
  )

  pred_clean <- predict(fit_clean, newdata = X, type = "link")
  expect_equal(predict(fit_guarded, newdata = X, type = "link"), pred_clean, tolerance = 1e-10)
  expect_false(grepl("count:poisson", fit_guarded$save_config(), fixed = TRUE))

  fit_dirty <- fit_clean
  fit_dirty$set_param("objective", "count:poisson")
  expect_equal(predict(fit_dirty, newdata = X, type = "link"), pred_clean, tolerance = 1e-10)
})
