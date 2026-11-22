test_that("cv_gbpp executes cleanly for xgboost", {
  skip_if_not_installed("spatstat.geom")
  skip_if_no_python("xgboost")
  
  set.seed(42)
  X <- data.frame(x1 = runif(100), x2 = runif(100))
  y <- c(rep(1, 40), rep(0, 60))
  locations <- data.frame(x = runif(100), y = runif(100))
  w <- c(rep(0, 40), runif(60))
  
  data <- new_gbpp_data(X = X, y = y, locations = locations, quadrature_weight = w, type = "planar")
  
  res <- cv_gbpp(data, engine = "xgboost", params = list(eta = 0.1, max_depth = 2L), n_trees = 5L, folds = 3L)
  
  expect_s3_class(res, "gbpp_cv")
  expect_true(is.numeric(res$metrics$loglik_mean))
  expect_true(is.numeric(res$metrics$deviance_mean))
  expect_equal(length(res$folds), 3L)
  
  # Check test proportions
  # Train events should be roughly 2/3, test events 1/3
  # And dummy points should be 60 in all folds
  expect_equal(sum(res$folds[[1]]$y == 0), 60L)

  out <- utils::capture.output(ret <- print(res))
  expect_identical(ret, res)
  expect_true(any(grepl("Stochastic Thinning Cross-Validation Results", out)))

  expect_error(cv_gbpp(data, folds = 1L), "at least 2")
  expect_error(cv_gbpp(data, folds = 50L), "at least equal")
})

test_that("cv_gbpp works with LightGBM, CatBoost, and logistic objective", {
  skip_if_not_installed("spatstat.random")
  skip_if_no_python(c("lightgbm", "catboost"))

  set.seed(123)
  W <- spatstat.geom::square(1)
  pp <- spatstat.random::runifpoint(24, win = W)
  covs <- list(z = spatstat.geom::as.im(function(x, y) x + y, W = W))

  dat_logi <- prepare_gbpp(pp, covariates = covs, quadrature_method = "logistic", nd = 8L)

  cv_lgb <- cv_gbpp(
    dat_logi,
    engine = "lightgbm",
    objective = "logistic",
    n_trees = 5L,
    folds = 3L,
    seed = 7L,
    verbose = FALSE
  )
  expect_s3_class(cv_lgb, "gbpp_cv")
  expect_true(is.finite(cv_lgb$metrics$loglik_mean))
  expect_true(is.finite(cv_lgb$metrics$deviance_mean))

  cv_cb <- cv_gbpp(
    dat_logi,
    engine = "catboost",
    objective = "logistic",
    n_trees = 5L,
    folds = 3L,
    seed = 7L,
    verbose = FALSE
  )
  expect_s3_class(cv_cb, "gbpp_cv")
  expect_true(is.finite(cv_cb$metrics$loglik_mean))
})
