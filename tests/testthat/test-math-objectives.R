test_that("dummy intensity helper extracts rho and computes m / |domain| across planar & network", {
  skip_if_not_installed("spatstat.random")
  skip_if_not_installed("spatstat.data")

  W <- spatstat.geom::owin(c(0, 2), c(0, 3)) # area = 6
  y <- c(rep(1, 10), rep(0, 30))
  expect_equal(mlpp:::.dummy_intensity(y, W), 30 / 6)
  expect_equal(mlpp:::.dummy_intensity(y, rep(6 / 40, 40)), 30 / 6)

  data(simplenet, package = "spatstat.data")
  len <- spatstat.geom::volume(simplenet)
  expect_equal(mlpp:::.dummy_intensity(y, simplenet), 30 / len)

  expect_error(mlpp:::.dummy_intensity(c(1, 2), W))
  expect_error(mlpp:::.dummy_intensity(rep(1, 10), W), "At least one dummy point")

  pp <- spatstat.random::runifpoint(n = 5, win = spatstat.geom::square(1))
  Q_logi <- spatstat.geom::quadscheme.logi(pp, dummytype = "grid", nd = 10)
  delta <- mlpp:::.dummy_intensity(Q_logi)
  expect_equal(delta, rep(Q_logi$param$rho, length(spatstat.geom::is.data(Q_logi))))

  Q_bt <- spatstat.geom::quadscheme(pp, nd = 10)
  expect_error(mlpp:::.dummy_intensity(Q_bt), "Dummy-process intensity could not be extracted")
})

test_that("iterative Poisson and Logistic weight helper matches formulas", {
  expect_equal(mlpp:::.iter_weight(c(-1, 0, 1), c(0, 0, 0)), c(1, 1, 1))

  w_p <- mlpp:::.iter_weight(log(c(2, 1, 0.5)), rep(0.5, 3))
  exp_p <- 1 / (1 + c(2, 1, 0.5) * 0.5)
  exp_p <- exp_p / mean(exp_p)
  expect_equal(w_p, exp_p)

  expect_true(all(is.finite(mlpp:::.iter_weight(c(-100, 0, 100), c(1, 1, 1)))))
  expect_error(mlpp:::.iter_weight(log(2), -1), "weight is undefined")

  pred <- log(c(1, 2, 4))
  delta <- c(1, 2, 4)
  A <- c(0, 0.5, 1)
  w_l <- mlpp:::.iter_weight(pred, A, delta = delta)
  lam <- exp(pred)
  exp_l <- (lam + delta) / (delta * (1 + lam * A))
  exp_l <- exp_l / mean(exp_l)
  expect_equal(w_l, exp_l, tolerance = 1e-12)
})

test_that(".ref_deriv matches exact derivatives across all 6 objectives", {
  dat <- mlpp_data(
    X = data.frame(x1 = 1:4, x2 = 4:1),
    y = c(1, 0, 1, 0),
    locations = data.frame(x = 0:3, y = 0:3),
    quadrature_weight = c(0.2, 0.3, 0.4, 0.5),
    dummy_intensity = rep(100, 4),
    statistical_weight = c(2, 0.5, 1.5, 1),
    interaction_integral = c(0, 0.5, 1, 2)
  )

  # Poisson
  pred <- log(c(2, 3, 4, 5))
  res_p <- mlpp:::.ref_deriv(dat, pred, objective = "poisson")
  expect_equal(res_p$gradient, dat$quadrature_weight * exp(pred) - dat$y)
  expect_equal(res_p$hessian, dat$quadrature_weight * exp(pred))

  # Weighted Poisson
  res_wp <- mlpp:::.ref_deriv(dat, pred, objective = "weighted_poisson")
  expect_equal(res_wp$gradient, dat$statistical_weight * (dat$quadrature_weight * exp(pred) - dat$y))
  expect_equal(res_wp$hessian, dat$statistical_weight * dat$quadrature_weight * exp(pred))

  # Logistic
  pred_logi <- rep(log(100), 4)
  res_l <- mlpp:::.ref_deriv(dat, pred_logi, objective = "logistic")
  expect_equal(res_l$gradient, rep(0.5, 4) - dat$y)
  expect_equal(res_l$hessian, rep(0.25, 4))

  # Weighted Logistic
  res_wl <- mlpp:::.ref_deriv(dat, pred_logi, objective = "weighted_logistic")
  expect_equal(res_wl$gradient, dat$statistical_weight * (0.5 - dat$y))
  expect_equal(res_wl$hessian, dat$statistical_weight * 0.25)

  # Iterative Poisson & Iterative Logistic
  res_ip <- mlpp:::.ref_deriv(dat, c(0, 0, 0, 0), objective = "iterative_poisson")
  w_ip <- mlpp:::.iter_weight(c(0, 0, 0, 0), dat$interaction_integral, clamp = TRUE)
  expect_equal(res_ip$weight, w_ip, tolerance = 1e-12)
  expect_equal(res_ip$gradient, w_ip * (dat$quadrature_weight - dat$y), tolerance = 1e-12)
  expect_equal(res_ip$hessian, w_ip * dat$quadrature_weight, tolerance = 1e-12)

  pred_il <- c(-0.5, 0, 0.5, 1)
  res_il <- mlpp:::.ref_deriv(dat, pred_il, objective = "iterative_logistic")
  w_il <- mlpp:::.iter_weight(pred_il, dat$interaction_integral, delta = dat$dummy_intensity, clamp = TRUE)
  pi_il <- stats::plogis(pred_il - log(dat$dummy_intensity))
  expect_equal(res_il$weight, w_il, tolerance = 1e-12)
  expect_equal(res_il$gradient, w_il * (pi_il - dat$y), tolerance = 1e-12)
  expect_equal(res_il$hessian, w_il * pi_il * (1 - pi_il), tolerance = 1e-12)
})

test_that("attached interaction integral supports iterative XGBoost fits", {
  skip_if_not_installed("spatstat.explore")
  skip_if_not_installed("spatstat.data")
  skip_if_no_python("xgboost")

  data(lansing, package = "spatstat.data")
  n <- 20L
  dat <- mlpp_data(
    X = matrix(rnorm(n * 2), ncol = 2),
    y = c(rep(1, 5), rep(0, 15)),
    locations = matrix(runif(n * 2), ncol = 2),
    quadrature_weight = rep(1 / n, n),
    dummy_intensity = rep(1, n),
    statistical_weight = rep(1, n)
  )
  dat <- attach_interaction(dat, rmax = 0.2, point_pattern = lansing[1:5], correction = "isotropic")

  for (obj in c("iterative_poisson", "iterative_logistic")) {
    fit <- fit_gbpp(dat, engine = "xgboost", objective = obj, params = list(eta = 0.05, max_depth = 2L), n_trees = 5L)
    expect_s3_class(fit, "mlpp_model")
    expect_true(all(is.finite(predict(fit, dat$X, type = "link"))))
  }
})
