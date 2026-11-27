test_that("plot.mlpp_model works for planar im and un-truncated link scale", {
  skip_if_no_python("xgboost")

  data <- mlpp_data(
    X = data.frame(x = c(0.5, 1.5, 2.5), y = c(0.5, 1.5, 2.5), v1 = c(1, 3, 5)),
    y = c(1, 0, 0),
    locations = data.frame(x = c(0.5, 1.5, 2.5), y = c(0.5, 1.5, 2.5)),
    quadrature_weight = c(1, 1, 1)
  )

  model <- fit_gbpp(
    data,
    engine = "xgboost",
    objective = "poisson",
    params = list(eta = 0.1, max_depth = 2L),
    n_trees = 2L
  )

  win <- spatstat.geom::owin(c(0, 3), c(0, 3))
  cov1 <- spatstat.geom::as.im(function(x, y) x + y, W = win, eps = 1)
  pts <- spatstat.geom::ppp(x = 1, y = 1, window = win)

  grDevices::pdf(file = NULL)
  on.exit(
    if (grDevices::dev.cur() > 1L) grDevices::dev.off(),
    add = TRUE
  )

  p_int <- plot(model, newdata = list(v1 = cov1), type = "intensity")
  expect_s3_class(p_int, "im")

  p_lnk <- plot(model, newdata = list(v1 = cov1), type = "link", overlay_points = TRUE, pts = pts)
  expect_s3_class(p_lnk, "im")
  expect_equal(p_lnk$v, log(p_int$v), tolerance = 1e-6)
  expect_error(plot(model), "must be supplied")

  # Large negative baseline (-60) should NOT be clamped when type = "link"
  model_extreme <- fit_gbpp(
    data,
    engine = "xgboost",
    objective = "poisson",
    params = list(eta = 0.1, max_depth = 2L),
    n_trees = 2L,
    baseline = -60.0
  )
  p_ext <- plot(model_extreme, newdata = list(v1 = cov1), type = "link")
  expect_true(all(p_ext$v[!is.na(p_ext$v)] < -55))
})

test_that("plot.mlpp_model renders network linim surfaces and overlays lpp events", {
  skip_if_not_installed("spatstat.data")
  skip_if_no_python("xgboost")

  set.seed(123)
  data(simplenet, package = "spatstat.data")
  pts <- spatstat.linnet::runiflpp(12, L = simplenet)
  lcov <- list(cx = spatstat.linnet::linfun(function(x, y, seg, tp) x + y, L = simplenet))
  dat <- prepare_mlpp(pts, covariates = lcov, nd = 10L)
  fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 4L, verbose = FALSE)

  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit({
    if (grDevices::dev.cur() > 1L) grDevices::dev.off()
    unlink(tmp_pdf)
  }, add = TRUE)

  p_net <- plot(fit, newdata = lcov, type = "intensity", overlay_points = TRUE, pts = pts)
  expect_s3_class(p_net, "linim")
  expect_warning(plot(fit, newdata = lcov, type = "link", overlay_points = TRUE), "must be provided")
})

test_that("plot.mlpp_cv renders separate and overlaid panels with and without window", {
  win <- spatstat.geom::square(1)
  cv_with_win <- structure(
    list(
      metrics = list(loglik_mean = -1.0, loglik_sd = 0.1, deviance_mean = 2.0, deviance_sd = 0.2),
      folds = list(
        list(y = c(1, 0), locations = data.frame(x = c(0.2, 0.5), y = c(0.3, 0.5)), window = win),
        list(y = c(1, 0), locations = data.frame(x = c(0.8, 0.5), y = c(0.7, 0.5)), window = win)
      ),
      type = "planar"
    ),
    class = "mlpp_cv"
  )
  cv_no_win <- cv_with_win
  cv_no_win$folds[[1]]$window <- NULL
  cv_no_win$folds[[2]]$window <- NULL

  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit({
    if (grDevices::dev.cur() > 1L) grDevices::dev.off()
    unlink(tmp_pdf)
  }, add = TRUE)

  expect_invisible(plot(cv_with_win, separate = TRUE))
  expect_invisible(plot(cv_with_win, separate = FALSE))
  expect_invisible(plot(cv_no_win, separate = FALSE))
})
