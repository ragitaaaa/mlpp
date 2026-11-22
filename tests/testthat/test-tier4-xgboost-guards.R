test_that("fit_gbpp strips params$objective and predict enforces output_margin for xgboost", {
  skip_if_not_installed("spatstat.geom")

  set.seed(123)
  win <- spatstat.geom::owin(c(0, 10), c(0, 10))
  n_pts <- 40L
  n_dum <- 160L
  pts <- spatstat.geom::ppp(
    x = runif(n_pts + n_dum, 0, 10),
    y = runif(n_pts + n_dum, 0, 10),
    window = win
  )
  X <- data.frame(
    x1 = rnorm(n_pts + n_dum),
    f1 = factor(sample(c("a", "b", "c"), n_pts + n_dum, replace = TRUE))
  )
  y <- c(rep(1L, n_pts), rep(0L, n_dum))
  w <- rep(100 / n_dum, n_pts + n_dum)

  dat <- new_gbpp_data(
    X = X,
    y = y,
    locations = pts,
    quadrature_weight = w,
    window = win,
    type = "planar"
  )

  fit_clean <- fit_gbpp(
    dat,
    engine = "xgboost",
    n_trees = 15L,
    learning_rate = 0.1,
    max_depth = 3L,
    random_seed = 42L
  )

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
  pred_guarded <- predict(fit_guarded, newdata = X, type = "link")
  expect_equal(pred_guarded, pred_clean, tolerance = 1e-10)
  expect_false(grepl("count:poisson", fit_guarded$save_config(), fixed = TRUE))

  # Even if an XGBoost booster carries a count:poisson objective attribute,
  # predict.gbpp_model() must request raw margins (output_margin = TRUE).
  fit_dirty <- fit_clean
  fit_dirty$set_param("objective", "count:poisson")
  pred_dirty <- predict(fit_dirty, newdata = X, type = "link")
  expect_equal(pred_dirty, pred_clean, tolerance = 1e-10)
})
