test_that("prepare_quad_data converts a planar quadrature scheme", {
  skip_if_not_installed("spatstat.random")

  W <- spatstat.geom::owin(
    xrange = c(0, 1),
    yrange = c(0, 1)
  )

  pp <- spatstat.random::runifpoint(
    n = 5,
    win = W
  )

  Q <- spatstat.geom::quadscheme(
    pp,
    nd = 10
  )

  X <- data.frame(
    x1 = seq_len(length(spatstat.geom::is.data(Q))),
    x2 = 1
  )

  data <- prepare_quad_data(
    Q = Q,
    X = X,
    objective = "poisson",
    type = "planar"
  )

  expect_s3_class(data, "gbpp_data")

  expect_equal(
    nrow(data$X),
    length(data$y)
  )

  expect_equal(
    length(data$quadrature_weight),
    length(data$y)
  )

  expect_equal(
    nrow(data$locations),
    length(data$y)
  )

  expect_equal(
    data$y,
    as.integer(spatstat.geom::is.data(Q))
  )

  expect_equal(
    data$type,
    "planar"
  )
})



test_that("prepare_quad_data extracts dummy intensity for logistic objective", {
  skip_if_not_installed("spatstat.random")

  W <- spatstat.geom::owin(
    xrange = c(0, 1),
    yrange = c(0, 1)
  )

  pp <- spatstat.random::runifpoint(
    n = 5,
    win = W
  )

  Q <- spatstat.geom::quadscheme.logi(
    pp,
    dummytype = "grid",
    nd = 10
  )

  X <- data.frame(
    x1 = seq_len(length(spatstat.geom::is.data(Q)))
  )

  data <- prepare_quad_data(
    Q = Q,
    X = X,
    objective = "logistic"
  )

  expect_type(data$dummy_intensity, "double")
  expect_length(
    data$dummy_intensity,
    length(data$y)
  )

  expect_equal(
    data$dummy_intensity,
    rep(Q$param$rho, length(data$y))
  )
})

test_that("prepare_quad_data does not create dummy intensity for Poisson objective", {
  skip_if_not_installed("spatstat.random")

  W <- spatstat.geom::owin(
    xrange = c(0, 1),
    yrange = c(0, 1)
  )

  pp <- spatstat.random::runifpoint(
    n = 5,
    win = W
  )

  Q <- spatstat.geom::quadscheme(
    pp,
    nd = 10
  )

  X <- data.frame(
    x1 = seq_len(length(spatstat.geom::is.data(Q)))
  )

  data <- prepare_quad_data(
    Q = Q,
    X = X,
    objective = "poisson"
  )

  expect_null(data$dummy_intensity)
})

test_that(
  "prepare_gbpp accepts covariate images that cover the point-pattern window",
  {
    skip_if_not_installed("spatstat.data")
    data(bei, package = "spatstat.data")

    covariate <- bei.extra$elev

    expect_false(
      identical(
        spatstat.geom::Window(bei),
        spatstat.geom::Window(covariate)
      )
    )

    expect_true(
      spatstat.geom::is.subset.owin(
        spatstat.geom::Window(bei),
        spatstat.geom::Window(covariate)
      )
    )

    data <- prepare_gbpp(
      point_pattern = bei,
      covariates = list(elevation = covariate),
      nd = 20L
    )

    expect_s3_class(data, "gbpp_data")

    expect_equal(
      nrow(data$X),
      length(data$y)
    )

    expect_identical(
      colnames(data$X),
      "elevation"
    )

    expect_true(
      all(is.finite(data$X$elevation))
    )

    expect_equal(
      data$type,
      "planar"
    )
  }
)

test_that("prepare_gbpp defaults to Berman-Turner quadrature with exact counting weights", {
  skip_if_not_installed("spatstat.random")
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(10, win = W)
  
  data <- prepare_gbpp(
    point_pattern = pp,
    nd = 10L
  )
  
  expect_s3_class(data, "gbpp_data")
  expect_equal(data$metadata$quadrature_method, "berman_turner")
  expect_equal(sum(data$quadrature_weight), 1.0, tolerance = 1e-6)
  expect_equal(sum(data$y == 1), 10L)
  expect_true(sum(data$y == 0) > 0)
  expect_type(data$dummy_intensity, "double")
  expect_equal(length(data$dummy_intensity), length(data$y))
  expect_equal(data$dummy_intensity[1], sum(data$y == 0) / 1.0)
})

test_that("prepare_gbpp supports logistic quadrature scheme with grid and poisson dummy points", {
  skip_if_not_installed("spatstat.random")
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(10, win = W)
  
  # Grid dummy points
  data_grid <- prepare_gbpp(
    point_pattern = pp,
    nd = 10L,
    quadrature_method = "logistic",
    dummytype = "grid"
  )
  
  expect_equal(data_grid$metadata$quadrature_method, "logistic")
  expect_equal(data_grid$metadata$dummytype, "grid")
  expect_type(data_grid$dummy_intensity, "double")
  expect_equal(data_grid$dummy_intensity[1], 100) # nd^2 = 100
  
  # Poisson random dummy points
  data_pois <- prepare_gbpp(
    point_pattern = pp,
    nd = 10L,
    quadrature_method = "logistic",
    dummytype = "poisson"
  )
  
  expect_equal(data_pois$metadata$quadrature_method, "logistic")
  expect_equal(data_pois$metadata$dummytype, "poisson")
  expect_type(data_pois$dummy_intensity, "double")
  expect_equal(data_pois$dummy_intensity[1], 100)
})

test_that("prepare_gbpp auto-selects quadrature scheme from objective", {
  skip_if_not_installed("spatstat.random")
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(10, win = W)
  
  data_poiss <- prepare_gbpp(pp, nd = 10L, objective = "poisson")
  expect_equal(data_poiss$metadata$quadrature_method, "berman_turner")
  
  data_logi <- prepare_gbpp(pp, nd = 10L, objective = "logistic")
  expect_equal(data_logi$metadata$quadrature_method, "logistic")
})

test_that("prepare_quad_data supports iterative_poisson and iterative_logistic", {
  skip_if_not_installed("spatstat.random")
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(5, win = W)
  
  Q_bt <- spatstat.geom::quadscheme(pp, nd = 10)
  X <- data.frame(x1 = seq_len(length(spatstat.geom::is.data(Q_bt))))
  
  data_iter_p <- prepare_quad_data(
    Q = Q_bt,
    X = X,
    objective = "iterative_poisson"
  )
  expect_s3_class(data_iter_p, "gbpp_data")
  expect_null(data_iter_p$dummy_intensity)
  
  Q_logi <- spatstat.geom::quadscheme.logi(pp, nd = 10)
  X_logi <- data.frame(x1 = seq_len(length(spatstat.geom::is.data(Q_logi))))
  data_iter_l <- prepare_quad_data(
    Q = Q_logi,
    X = X_logi,
    objective = "iterative_logistic"
  )
  expect_s3_class(data_iter_l, "gbpp_data")
  expect_type(data_iter_l$dummy_intensity, "double")
})

test_that("prepare_gbpp builds full planar gbpp_data and preserves factor im covariates", {
  skip_if_not_installed("spatstat.random")

  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(10, win = W)
  elev <- spatstat.geom::as.im(function(x, y) x + y, W = W)
  zone <- spatstat.geom::as.im(
    function(x, y) factor(ifelse(x < 0.5, "west", "east"), levels = c("west", "east")),
    W = W
  )

  dat <- prepare_gbpp(pp, covariates = list(elevation = elev, zone = zone), nd = 10L)
  expect_s3_class(dat, "gbpp_data")
  expect_equal(dat$type, "planar")
  expect_identical(names(dat$X), c("elevation", "zone"))
  expect_true(is.factor(dat$X$zone) || is.character(dat$X$zone))
  expect_identical(dat$window, W)
})

test_that("prepare_gbpp validates point_pattern, covariates, window coverage, and nd", {
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  W_small <- spatstat.geom::owin(c(0, 0.5), c(0, 0.5))
  pp <- spatstat.geom::ppp(x = c(0.2, 0.8), y = c(0.2, 0.8), window = W)
  im_ok <- spatstat.geom::as.im(function(x, y) x, W = W)
  im_small <- spatstat.geom::as.im(function(x, y) x, W = W_small)
  im_na <- spatstat.geom::as.im(function(x, y) ifelse(x > 0.5, NA_real_, x), W = W)

  expect_error(prepare_gbpp(list()), "ppp")
  expect_error(prepare_gbpp(pp, covariates = unname(list(im_ok)), nd = 5L), "named")
  expect_error(prepare_gbpp(pp, covariates = list(a = im_ok, a = im_ok), nd = 5L), "unique")
  expect_error(prepare_gbpp(pp, covariates = list(a = 1:5), nd = 5L), "im")
  expect_error(prepare_gbpp(pp, covariates = list(a = im_small), nd = 5L), "cover")
  expect_error(prepare_gbpp(pp, covariates = list(a = im_na), nd = 5L), "cover|missing")
  expect_error(prepare_gbpp(pp, nd = 0L), "positive integer")
})

