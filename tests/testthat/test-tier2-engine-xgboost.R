test_that("XGBoost custom objectives match R reference derivatives across all 6 objectives", {
  skip_if_no_python("xgboost")

  data <- make_small_gbpp_data()
  prediction <- seq(-0.6, 0.8, length.out = nrow(data$X))

  xgb <- reticulate::import("xgboost", delay_load = FALSE)
  dtrain <- xgb$DMatrix(data = as.matrix(data$X), label = as.numeric(data$y))

  objectives <- c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )

  for (obj_name in objectives) {
    expected <- calc_objective(data = data, prediction = prediction, objective = obj_name)
    py_obj <- mlpp:::.gbpp_make_objective(
      engine = "xgboost",
      objective = obj_name,
      quadrature_weight = data$quadrature_weight,
      dummy_intensity = data$dummy_intensity,
      statistical_weight = data$statistical_weight,
      interaction_integral = data$interaction_integral
    )

    res <- py_obj(reticulate::r_to_py(prediction), dtrain)
    grad <- as.numeric(reticulate::py_to_r(res[[1]]))
    hess <- as.numeric(reticulate::py_to_r(res[[2]]))

    expect_equal(grad, expected$gradient, tolerance = 1e-12, info = paste("grad:", obj_name))
    expect_equal(hess, expected$hessian, tolerance = 1e-12, info = paste("hess:", obj_name))
  }
})

test_that("XGBoost fits and predicts cleanly for all 6 objectives (data.frame and matrix)", {
  skip_if_no_python("xgboost")

  data <- make_small_gbpp_data()
  objectives <- c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )

  for (obj_name in objectives) {
    fit <- fit_gbpp(
      data = data,
      engine = "xgboost",
      objective = obj_name,
      params = list(eta = 0.1, max_depth = 2L),
      n_trees = 5L
    )

    expect_s3_class(fit, "gbpp_xgboost")
    expect_false(reticulate::py_is_null_xptr(fit))

    pred_mat <- predict(fit, newdata = data$X, type = "link")
    pred_df <- predict(fit, newdata = as.data.frame(data$X), type = "link")
    pred_int <- predict(fit, newdata = data$X, type = "intensity")

    expect_type(pred_mat, "double")
    expect_length(pred_mat, nrow(data$X))
    expect_true(all(is.finite(pred_mat)))
    expect_equal(pred_df, pred_mat, tolerance = 1e-12, info = paste("df vs matrix:", obj_name))
    expect_equal(pred_int, exp(pred_mat), tolerance = 1e-6, info = paste("exp(link):", obj_name))
  }
})

test_that("XGBoost model integrates with prepare_gbpp planar features", {
  skip_if_not_installed("spatstat.random")
  skip_if_no_python("xgboost")

  set.seed(123)
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(10, win = W)
  cov1 <- spatstat.geom::as.im(function(x, y) x + y, W = W)

  dat <- prepare_gbpp(pp, covariates = list(elevation = cov1), nd = 10L)
  fit <- fit_gbpp(
    data = dat,
    engine = "xgboost",
    objective = "poisson",
    params = list(eta = 0.1, max_depth = 2L),
    n_trees = 3L
  )

  pred <- predict(fit, newdata = dat$X, type = "intensity")
  expect_type(pred, "double")
  expect_length(pred, nrow(dat$X))
  expect_true(all(is.finite(pred)) && all(pred > 0))
})
