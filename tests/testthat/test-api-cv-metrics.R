test_that("eval_mlpp computes Poisson log-likelihood and deviance from vectors or model+data", {
  set.seed(42)
  pred <- runif(10, 0.1, 1.0)
  y <- c(1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
  v <- c(0, 0, 0, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1)

  ll_exp <- sum(log(pred[y == 1])) - sum(v * pred)
  v_safe <- v
  v_safe[y == 1] <- pmax(v_safe[y == 1], 1e-15)
  mu <- v_safe * pred
  dev_exp <- 2 * (sum(-log(mu[y == 1])) - sum(y - mu))

  both <- eval_mlpp(pred, y, v)
  expect_named(both, c("loglik", "deviance"))
  expect_equal(unname(both["loglik"]), ll_exp)
  expect_equal(unname(both["deviance"]), dev_exp)
  expect_equal(eval_mlpp(pred, y, v, metric = "loglik"), ll_exp)
  expect_equal(eval_mlpp(pred, y, v, metric = "deviance"), dev_exp)

  p <- 0.2
  ll_p <- sum(log(pred[y == 1] * p)) - sum(v * pred * p)
  expect_equal(eval_mlpp(pred, y, v, metric = "loglik", p_scale = p), ll_p)

  expect_error(eval_mlpp(c(1, NA), c(1, 0), c(0.5, 0.5)), "finite")
  expect_error(eval_mlpp(c(1, 2), c(2, 0), c(0.5, 0.5)), "0 \\(dummy\\) and 1 \\(event\\)")
  expect_error(eval_mlpp(c(1, 2), c(1, 0), c(-1, 1)), "non-negative")
  expect_error(eval_mlpp(c(1, 2), c(1, 0), c(0.5, 0.5), p_scale = 0), "p_scale")

  skip_if_no_python("xgboost")
  dat <- make_small_mlpp_data()
  fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 4L, max_depth = 2L)
  m_res <- eval_mlpp(fit, dat)
  pred_fit <- predict(fit, dat$X, type = "intensity")
  expect_equal(m_res, eval_mlpp(pred_fit, dat$y, dat$quadrature_weight), tolerance = 1e-10)
})

test_that("cv_mlpp executes cleanly across engines and conserves spatial measure + caller RNG", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  set.seed(505)
  W <- spatstat.geom::owin(c(0, 8), c(0, 6))
  pp <- spatstat.random::runifpoint(25, win = W)
  covs <- list(z1 = spatstat.geom::as.im(function(x, y) x + 0.2 * y, W = W))
  dat_p <- prepare_mlpp(pp, covariates = covs, nd = 8L)

  set.seed(999)
  r1 <- stats::runif(1)
  set.seed(999)
  seed_before <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)

  cv_p <- cv_mlpp(dat_p, engine = "xgboost", n_trees = 5L, folds = 3L, seed = 42L, verbose = FALSE)
  expect_identical(get(".Random.seed", envir = .GlobalEnv, inherits = FALSE), seed_before)
  expect_equal(stats::runif(1), r1)

  expect_s3_class(cv_p, "mlpp_cv")
  expect_length(cv_p$folds, 3L)
  expect_true(is.numeric(cv_p$metrics$loglik_mean))
  expect_true(is.numeric(cv_p$metrics$deviance_mean))
  for (k in seq_len(3L)) {
    expect_equal(cv_p$folds[[k]]$weight_sum, spatstat.geom::area(W), tolerance = 1e-6)
    expect_equal(cv_p$folds[[k]]$train_weight_sum, spatstat.geom::area(W), tolerance = 1e-6)
  }

  out <- utils::capture.output(ret <- print(cv_p))
  expect_identical(ret, cv_p)
  expect_true(any(grepl("Stochastic Thinning Cross-Validation Results", out)))

  # Logistic CV with LightGBM and CatBoost
  dat_logi <- prepare_mlpp(pp, covariates = covs, quadrature_method = "logistic", nd = 8L)
  cv_lgb <- cv_mlpp(dat_logi, engine = "lightgbm", objective = "logistic", n_trees = 5L, folds = 3L, seed = 7L, verbose = FALSE)
  cv_cb <- cv_mlpp(dat_logi, engine = "catboost", objective = "logistic", n_trees = 5L, folds = 3L, seed = 7L, verbose = FALSE)
  expect_true(is.finite(cv_lgb$metrics$loglik_mean))
  expect_true(is.finite(cv_cb$metrics$loglik_mean))

  # Network measure conservation
  data(simplenet, package = "spatstat.data")
  lp <- spatstat.linnet::runiflpp(20, L = simplenet)
  lcov <- list(z1 = spatstat.linnet::linfun(function(x, y, seg, tp) x + seg, L = simplenet))
  dat_net <- prepare_mlpp(lp, covariates = lcov, nd = 10L)
  cv_n <- cv_mlpp(dat_net, engine = "xgboost", n_trees = 5L, folds = 3L, seed = 42L, verbose = FALSE)
  net_len <- spatstat.geom::volume(simplenet)
  for (k in seq_len(3L)) {
    expect_equal(cv_n$folds[[k]]$weight_sum, net_len, tolerance = 1e-6)
    expect_equal(cv_n$folds[[k]]$train_weight_sum, net_len, tolerance = 1e-6)
  }

  expect_error(cv_mlpp(dat_p, folds = 1L), "at least 2")
  expect_error(cv_mlpp(dat_p, folds = 50L), "at least equal")
})
