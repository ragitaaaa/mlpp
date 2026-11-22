test_that("predict(type = 'im') works correctly", {
  skip_if_no_python("xgboost")

  data <- new_gbpp_data(
    X = data.frame(
      x = c(0.5, 1.5, 2.5),
      y = c(0.5, 1.5, 2.5),
      v1 = c(1, 3, 5)
    ),
    y = c(1, 0, 0),
    locations = data.frame(
      x = c(0.5, 1.5, 2.5),
      y = c(0.5, 1.5, 2.5)
    ),
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

  expect_error(
    predict(model, newdata = win, type = "im"),
    "Missing required covariates in `newdata`: v1"
  )

  pred_cov <- predict(model, newdata = list(v1 = cov1), type = "im")
  expect_s3_class(pred_cov, "im")
  expect_equal(pred_cov$dim, c(3, 3))
  expect_true(all(is.finite(pred_cov$v[!is.na(pred_cov$v)])))

  # Test with eps param
  pred_cov2 <- predict(model, newdata = list(v1 = cov1), type = "im", eps = 0.5)
  expect_s3_class(pred_cov2, "im")
  expect_equal(pred_cov2$dim, c(6, 6))
})

test_that("predict(type = 'linim') works correctly", {
  skip_if_not_installed("spatstat.data")
  skip_if_no_python("xgboost")

  L <- spatstat.data::simplenet
  pts <- spatstat.linnet::rpoislpp(5, L)

  df_X <- data.frame(
    x = as.numeric(pts$data$x),
    y = as.numeric(pts$data$y),
    v1 = runif(spatstat.geom::npoints(pts))
  )

  data <- new_gbpp_data(
    X = df_X,
    y = rep(1, spatstat.geom::npoints(pts)),
    locations = df_X[, c("x", "y")],
    quadrature_weight = rep(1, spatstat.geom::npoints(pts))
  )

  model <- fit_gbpp(
    data,
    engine = "xgboost",
    objective = "poisson",
    params = list(eta = 0.1, max_depth = 2L),
    n_trees = 2L
  )

  cov1 <- spatstat.linnet::as.linim(
    spatstat.linnet::linfun(function(x, y, seg, tp) x + y, L)
  )

  expect_error(
    predict(model, newdata = L, type = "linim"),
    "Missing required covariates in `newdata`: v1"
  )

  pred_cov <- predict(model, newdata = list(v1 = cov1), type = "linim")
  expect_s3_class(pred_cov, "linim")
  expect_true(is.numeric(pred_cov$v))
  expect_true(all(is.finite(pred_cov$v[!is.na(pred_cov$v)])))

  # Test with eps param
  pred_cov2 <- predict(model, newdata = list(v1 = cov1), type = "linim", eps = 0.5)
  expect_s3_class(pred_cov2, "linim")
})

test_that("plot.gbpp_model works for planar im", {
  skip_if_no_python("xgboost")

  data <- new_gbpp_data(
    X = data.frame(
      x = c(0.5, 1.5, 2.5),
      y = c(0.5, 1.5, 2.5),
      v1 = c(1, 3, 5)
    ),
    y = c(1, 0, 0),
    locations = data.frame(
      x = c(0.5, 1.5, 2.5),
      y = c(0.5, 1.5, 2.5)
    ),
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

  p1 <- plot(model, newdata = list(v1 = cov1), type = "intensity")
  expect_s3_class(p1, "im")

  p2 <- plot(model, newdata = list(v1 = cov1), type = "link", overlay_points = TRUE, pts = pts)
  expect_s3_class(p2, "im")

  expect_equal(p2$v, log(p1$v), tolerance = 1e-6)
  expect_error(plot(model), "must be supplied")
})

test_that("plot.gbpp_model renders network linim surfaces and overlays lpp events", {
  skip_if_not_installed("spatstat.data")
  skip_if_no_python("xgboost")

  set.seed(123)
  data(simplenet, package = "spatstat.data")
  pts <- spatstat.linnet::runiflpp(12, L = simplenet)
  lcov <- list(cx = spatstat.linnet::linfun(function(x, y, seg, tp) x + y, L = simplenet))
  dat <- prepare_gbpp_network(pts, covariates = lcov, nd = 10L)
  fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 4L, verbose = FALSE)

  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit(
    {
      if (grDevices::dev.cur() > 1L) grDevices::dev.off()
      unlink(tmp_pdf)
    },
    add = TRUE
  )

  p_net <- plot(fit, newdata = lcov, type = "intensity", overlay_points = TRUE, pts = pts)
  expect_s3_class(p_net, "linim")
  expect_warning(
    plot(fit, newdata = lcov, type = "link", overlay_points = TRUE),
    "must be provided"
  )
})
