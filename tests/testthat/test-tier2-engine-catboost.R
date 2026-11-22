test_that("CatBoost custom objectives match R reference derivatives (stateful and stateless)", {
  skip_if_no_python("catboost")

  data <- make_small_gbpp_data()
  data$statistical_weight <- c(1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.0)
  prediction <- seq(-0.6, 0.8, length.out = nrow(data$X))

  extract_catboost_ders <- function(py_obj, approx, targets, weights = NULL) {
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

  objectives <- c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )

  # 1. Stateful objective classes (calc_ders_range returns (-g, -h))
  for (obj_name in objectives) {
    expected <- calc_objective(data = data, prediction = prediction, objective = obj_name)
    py_obj <- mlpp:::.gbpp_make_objective(
      engine = "catboost",
      objective = obj_name,
      quadrature_weight = data$quadrature_weight,
      dummy_intensity = data$dummy_intensity,
      statistical_weight = data$statistical_weight,
      interaction_integral = data$interaction_integral,
      stateless = FALSE
    )

    got <- extract_catboost_ders(py_obj, prediction, data$y)
    expect_equal(got$gradient, expected$gradient, tolerance = 1e-12, info = paste("stateful grad:", obj_name))
    expect_equal(got$hessian, expected$hessian, tolerance = 1e-12, info = paste("stateful hess:", obj_name))
  }

  # 2. Stateless objectives (baseline-shifted Pool inputs + optional Pool weights)
  for (obj_name in c("poisson", "weighted_poisson", "logistic", "weighted_logistic")) {
    expected <- calc_objective(data = data, prediction = prediction, objective = obj_name)
    py_obj <- mlpp:::.gbpp_make_objective(
      engine = "catboost",
      objective = obj_name,
      stateless = TRUE
    )
    approx_shifted <- if (grepl("poisson", obj_name)) {
      prediction + log(data$quadrature_weight)
    } else {
      prediction - log(data$dummy_intensity)
    }
    pool_w <- if (grepl("weighted", obj_name)) data$statistical_weight else NULL

    got <- extract_catboost_ders(py_obj, approx_shifted, data$y, weights = pool_w)
    expect_equal(got$gradient, expected$gradient, tolerance = 1e-12, info = paste("stateless grad:", obj_name))
    expect_equal(got$hessian, expected$hessian, tolerance = 1e-12, info = paste("stateless hess:", obj_name))
  }
})

test_that("CatBoost backend fits and predicts all GBPP objectives", {
  skip_if_no_python("catboost")

  data <- make_small_gbpp_data()
  params <- list(
    depth = 2L,
    learning_rate = 0.05,
    verbose = FALSE,
    random_seed = 123L,
    thread_count = 1L,
    eval_metric = "RMSE"
  )

  objectives <- c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )

  for (obj_name in objectives) {
    model <- fit_gbpp(
      data = data,
      engine = "catboost",
      objective = obj_name,
      params = params,
      n_trees = 10L
    )

    expect_s3_class(model, "gbpp_catboost")
    prediction <- predict(model, newdata = data$X)

    expect_type(prediction, "double")
    expect_length(prediction, nrow(data$X))
    expect_true(all(is.finite(prediction)))
  }
})

test_that("CatBoost defaults to one_hot_max_size = 0 (always OTS) and supports Ordered boosting", {
  skip_if_no_python("catboost")

  set.seed(11)
  n <- 40L
  dat <- new_gbpp_data(
    X = data.frame(
      x1 = stats::rnorm(n),
      road_bin = factor(sample(c("Arteri", "Lokal"), n, replace = TRUE), levels = c("Arteri", "Lokal"))
    ),
    y = as.numeric(stats::runif(n) < 0.3),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = stats::runif(n, 0.5, 1.5),
    type = "planar"
  )

  fit_default <- fit_gbpp(
    dat,
    engine = "catboost",
    objective = "poisson",
    n_trees = 10L,
    max_depth = 3L,
    verbose = FALSE
  )
  expect_equal(as.integer(fit_default$get_param("one_hot_max_size")), 0L)

  fit_ordered <- fit_gbpp(
    dat,
    engine = "catboost",
    objective = "poisson",
    n_trees = 10L,
    max_depth = 3L,
    params = list(boosting_type = "Ordered"),
    verbose = FALSE
  )
  expect_equal(as.integer(fit_ordered$get_param("one_hot_max_size")), 0L)
  expect_equal(as.character(fit_ordered$get_param("boosting_type")), "Ordered")
  expect_true(all(is.finite(predict(fit_ordered, dat$X, type = "link"))))
})
