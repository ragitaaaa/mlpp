test_that("fit_gbpp fits across all engines with unified API", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  data <- make_small_mlpp_data()

  m_xgb <- fit_gbpp(
    data = data,
    engine = "xgboost",
    objective = "poisson",
    params = list(max_depth = 2L, learning_rate = 0.1),
    n_trees = 5L
  )
  expect_s3_class(m_xgb, "mlpp_model")
  expect_s3_class(m_xgb, "gbpp_model")
  expect_s3_class(m_xgb, "gbpp_xgboost")
  expect_equal(attr(m_xgb, "engine"), "xgboost")
  expect_equal(attr(m_xgb, "objective"), "poisson")

  m_lgb <- fit_gbpp(
    data = data,
    engine = "lightgbm",
    objective = "poisson",
    params = list(max_depth = 2L, learning_rate = 0.1),
    n_trees = 5L
  )
  expect_s3_class(m_lgb, "mlpp_model")
  expect_s3_class(m_lgb, "gbpp_lightgbm")
  expect_equal(attr(m_lgb, "engine"), "lightgbm")

  m_cat <- fit_gbpp(
    data = data,
    engine = "catboost",
    objective = "poisson",
    params = list(depth = 2L, learning_rate = 0.1),
    n_trees = 5L
  )
  expect_s3_class(m_cat, "mlpp_model")
  expect_s3_class(m_cat, "gbpp_catboost")
  expect_equal(attr(m_cat, "engine"), "catboost")
})

test_that("fit_gbpp fits and predicts all 6 objectives across engines", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  data <- make_small_mlpp_data()
  objectives <- c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    extra_params <- if (eng == "lightgbm") {
      list(min_data_in_leaf = 1L, feature_pre_filter = FALSE)
    } else if (eng == "catboost") {
      list(depth = 2L)
    } else {
      list(max_depth = 2L, eta = 0.1)
    }

    for (obj_name in objectives) {
      fit <- fit_gbpp(
        data = data,
        engine = eng,
        objective = obj_name,
        params = extra_params,
        n_trees = 5L
      )
      p_link <- predict(fit, newdata = data$X, type = "link")
      p_int <- predict(fit, newdata = data$X, type = "intensity")
      expect_length(p_link, nrow(data$X))
      expect_true(all(is.finite(p_link)))
      expect_equal(p_int, exp(p_link), tolerance = 1e-6)
    }
  }
})

test_that("CatBoost defaults to one_hot_max_size = 0 and supports Ordered boosting", {
  skip_if_no_python("catboost")

  set.seed(11)
  n <- 40L
  dat <- mlpp_data(
    X = data.frame(
      x1 = stats::rnorm(n),
      road_bin = factor(sample(c("Arteri", "Lokal"), n, replace = TRUE), levels = c("Arteri", "Lokal"))
    ),
    y = as.numeric(stats::runif(n) < 0.3),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = stats::runif(n, 0.5, 1.5),
    type = "planar"
  )

  fit_default <- fit_gbpp(dat, engine = "catboost", objective = "poisson", n_trees = 10L, max_depth = 3L, verbose = FALSE)
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
  expect_equal(as.character(fit_ordered$get_param("boosting_type")), "Ordered")
  expect_true(all(is.finite(predict(fit_ordered, dat$X, type = "link"))))
})

test_that("predict.mlpp_model respects type contract: link, intensity, raw, im, linim", {
  skip_if_not_installed("spatstat.data")
  skip_if_no_python("xgboost")

  data <- make_small_mlpp_data()
  m_xgb <- fit_gbpp(data = data, engine = "xgboost", objective = "poisson", n_trees = 5L)

  p_intensity <- predict(m_xgb, newdata = data$X, type = "intensity")
  p_link <- predict(m_xgb, newdata = data$X, type = "link")
  p_raw <- predict(m_xgb, newdata = data$X, type = "raw")

  expect_equal(p_intensity, exp(p_link), tolerance = 1e-6)
  expect_equal(p_link, p_raw + attr(m_xgb, "baseline"), tolerance = 1e-6)

  # Spatial im and linim
  dat_sp <- mlpp_data(
    X = data.frame(x = c(0.5, 1.5, 2.5), y = c(0.5, 1.5, 2.5), v1 = c(1, 3, 5)),
    y = c(1, 0, 0),
    locations = data.frame(x = c(0.5, 1.5, 2.5), y = c(0.5, 1.5, 2.5)),
    quadrature_weight = c(1, 1, 1)
  )
  m_sp <- fit_gbpp(dat_sp, engine = "xgboost", objective = "poisson", max_depth = 2L, n_trees = 3L)

  win <- spatstat.geom::owin(c(0, 3), c(0, 3))
  cov_im <- spatstat.geom::as.im(function(x, y) x + y, W = win, eps = 1)
  expect_error(predict(m_sp, newdata = win, type = "im"), "Missing required covariates")
  pred_im <- predict(m_sp, newdata = list(v1 = cov_im), type = "im")
  expect_s3_class(pred_im, "im")
  expect_equal(pred_im$dim, c(3, 3))

  L <- spatstat.data::simplenet
  cov_lin <- spatstat.linnet::as.linim(spatstat.linnet::linfun(function(x, y, seg, tp) x + y, L))
  expect_error(predict(m_sp, newdata = L, type = "linim"), "Missing required covariates")
  pred_lin <- predict(m_sp, newdata = list(v1 = cov_lin), type = "linim")
  expect_s3_class(pred_lin, "linim")
})

test_that("network pipelines work end-to-end across Poisson, Logistic, and Iterative objectives", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.linnet")
  skip_if_no_python("xgboost")

  set.seed(123)
  fixture <- make_network_integration_fixture(n = 30L)
  base_data <- prepare_mlpp(x = fixture$point_pattern, covariates = fixture$covariates, nd = 20L)

  data_fixed <- base_data
  data_fixed$interaction_integral <- rep(0.01, length(data_fixed$y))

  data_est <- attach_interaction(data = base_data, rmax = 0.005, nr = 32L)
  expect_length(data_est$interaction_integral, nrow(data_est$X))
  expect_true(all(is.finite(data_est$interaction_integral)))

  cases <- list(
    list(obj = "poisson", dat = base_data),
    list(obj = "logistic", dat = base_data),
    list(obj = "iterative_poisson", dat = data_fixed),
    list(obj = "iterative_logistic", dat = data_fixed),
    list(obj = "iterative_poisson", dat = data_est),
    list(obj = "iterative_logistic", dat = data_est)
  )

  for (case in cases) {
    model <- fit_gbpp(
      data = case$dat,
      engine = "xgboost",
      objective = case$obj,
      params = list(eta = 0.1, max_depth = 2L),
      n_trees = 5L
    )
    preds <- predict(model, newdata = case$dat$X, type = "intensity")
    expect_true(all(is.finite(preds)) && all(preds > 0), info = case$obj)
  }
})

test_that("fit_gbpp rejects invalid inputs and print.mlpp_model handles scalar and vector baselines", {
  data <- make_small_mlpp_data()
  expect_error(fit_gbpp(data = data, engine = "invalid_engine"))
  expect_error(fit_gbpp(data = list(not = "data"), engine = "xgboost"), "mlpp_data")

  skip_if_no_python("xgboost")
  fit_scalar <- fit_gbpp(data, engine = "xgboost", objective = "poisson", n_trees = 3L)
  out_s <- utils::capture.output(ret_s <- print(fit_scalar))
  expect_identical(ret_s, fit_scalar)
  expect_true(any(grepl("Baseline \\(eta0\\)", out_s)))

  fit_vec <- fit_gbpp(
    data,
    engine = "xgboost",
    objective = "poisson",
    n_trees = 3L,
    baseline = seq(-1.0, 0.0, length.out = nrow(data$X))
  )
  out_v <- utils::capture.output(print(fit_vec))
  expect_true(any(grepl("vector", out_v)))
})
