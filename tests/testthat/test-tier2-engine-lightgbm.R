test_that("LightGBM custom objectives match R reference derivatives across all 6 objectives", {
  skip_if_no_python("lightgbm")

  data <- make_small_gbpp_data()
  prediction <- seq(-0.6, 0.8, length.out = nrow(data$X))

  lgb <- reticulate::import("lightgbm", delay_load = FALSE)
  dtrain <- lgb$Dataset(data = as.matrix(data$X), label = as.numeric(data$y))
  dtrain$construct()

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
      engine = "lightgbm",
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

test_that("LightGBM fits and predicts cleanly for all 6 objectives", {
  skip_if_no_python("lightgbm")

  data <- make_small_gbpp_data()
  fit_params <- list(min_data_in_leaf = 1L, feature_pre_filter = FALSE)
  objectives <- c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )

  newdata <- data$X[1:3, , drop = FALSE]

  for (obj_name in objectives) {
    model <- fit_gbpp(
      data = data,
      engine = "lightgbm",
      objective = obj_name,
      params = fit_params,
      n_trees = 5L
    )

    expect_s3_class(model, "gbpp_lightgbm")
    prediction <- predict(model, newdata = newdata, type = "link")

    expect_type(prediction, "double")
    expect_length(prediction, nrow(newdata))
    expect_true(all(is.finite(prediction)))
  }
})

test_that("LightGBM model can predict from gbpp_data features", {
  skip_if_not_installed("spatstat.random")
  skip_if_no_python("lightgbm")

  set.seed(123)
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(n = 10, win = W)
  covariate <- spatstat.geom::as.im(function(x, y) x + y, W = W)

  data <- prepare_gbpp(
    point_pattern = pp,
    covariates = list(elevation = covariate),
    nd = 10L
  )

  model <- fit_gbpp(
    data = data,
    engine = "lightgbm",
    objective = "poisson",
    params = list(
      learning_rate = 0.1,
      max_depth = 2L,
      num_leaves = 4L,
      min_data_in_leaf = 1L,
      feature_pre_filter = FALSE
    ),
    n_trees = 3L
  )

  prediction <- predict(model, data$X)
  expect_type(prediction, "double")
  expect_length(prediction, nrow(data$X))
  expect_true(all(is.finite(prediction)))
})
