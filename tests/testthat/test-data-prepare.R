test_that("prepare_mlpp prepares planar Berman-Turner and logistic data cleanly", {
  skip_if_not_installed("spatstat.random")
  skip_if_not_installed("spatstat.data")

  data(bei, package = "spatstat.data")
  covariate <- bei.extra$elev
  dat_bei <- prepare_mlpp(x = bei, covariates = list(elevation = covariate), nd = 20L)
  expect_s3_class(dat_bei, "mlpp_data")
  expect_identical(dat_bei$type, "planar")
  expect_identical(colnames(dat_bei$X), "elevation")
  expect_true(all(is.finite(dat_bei$X$elevation)))

  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(10, win = W)

  dat_bt <- prepare_mlpp(pp, nd = 10L)
  expect_equal(dat_bt$metadata$quadrature_method, "berman_turner")
  expect_equal(sum(dat_bt$quadrature_weight), 1.0, tolerance = 1e-6)
  expect_equal(dat_bt$dummy_intensity[1], sum(dat_bt$y == 0) / 1.0)

  dat_grid <- prepare_mlpp(pp, nd = 10L, quadrature_method = "logistic", dummytype = "grid")
  expect_equal(dat_grid$metadata$quadrature_method, "logistic")
  expect_equal(dat_grid$metadata$dummytype, "grid")
  expect_equal(dat_grid$dummy_intensity[1], 100)

  dat_pois <- prepare_mlpp(pp, nd = 10L, quadrature_method = "logistic", dummytype = "poisson")
  expect_equal(dat_pois$metadata$dummytype, "poisson")
  expect_equal(dat_pois$dummy_intensity[1], 100)

  expect_equal(prepare_mlpp(pp, nd = 10L, objective = "poisson")$metadata$quadrature_method, "berman_turner")
  expect_equal(prepare_mlpp(pp, nd = 10L, objective = "logistic")$metadata$quadrature_method, "logistic")

  elev <- spatstat.geom::as.im(function(x, y) x + y, W = W)
  zone <- spatstat.geom::as.im(
    function(x, y) factor(ifelse(x < 0.5, "west", "east"), levels = c("west", "east")),
    W = W
  )
  dat_fac <- prepare_mlpp(pp, covariates = list(elevation = elev, zone = zone), nd = 10L, interaction_integral = 0.25)
  expect_true(is.factor(dat_fac$X$zone) || is.character(dat_fac$X$zone))
  expect_equal(dat_fac$interaction_integral, rep(0.25, length(dat_fac$y)))
})

test_that("prepare_mlpp validates planar inputs", {
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  W_small <- spatstat.geom::owin(c(0, 0.5), c(0, 0.5))
  pp <- spatstat.geom::ppp(x = c(0.2, 0.8), y = c(0.2, 0.8), window = W)
  im_ok <- spatstat.geom::as.im(function(x, y) x, W = W)
  im_small <- spatstat.geom::as.im(function(x, y) x, W = W_small)
  im_na <- spatstat.geom::as.im(function(x, y) ifelse(x > 0.5, NA_real_, x), W = W)

  expect_error(prepare_mlpp(list()), "`x` must be a `spatstat`")
  expect_error(prepare_mlpp(pp, covariates = unname(list(im_ok)), nd = 5L), "named")
  expect_error(prepare_mlpp(pp, covariates = list(a = im_ok, a = im_ok), nd = 5L), "unique")
  expect_error(prepare_mlpp(pp, covariates = list(a = 1:5), nd = 5L), "im")
  expect_error(prepare_mlpp(pp, covariates = list(a = im_small), nd = 5L), "cover")
  expect_error(prepare_mlpp(pp, covariates = list(a = im_na), nd = 5L), "cover|missing")
  expect_error(prepare_mlpp(pp, nd = 0L), "positive integer")
})

test_that("prepare_mlpp prepares linear network data and preserves segment topology", {
  skip_if_not_installed("spatstat.data")
  data(chicago, package = "spatstat.data")

  L <- spatstat.linnet::as.linnet(chicago)
  lf <- spatstat.linnet::linfun(function(x, y, seg, tp) x + y, L)
  l_im <- spatstat.linnet::as.linim(lf)

  out_f <- prepare_mlpp(chicago, covariates = list(xy_sum = lf), nd = 20L, interaction_integral = -0.1)
  expect_s3_class(out_f, "mlpp_data")
  expect_identical(out_f$type, "network")
  expect_s3_class(out_f$locations, "lpp")
  expect_equal(out_f$interaction_integral, rep(-0.1, length(out_f$y)))

  out_im <- prepare_mlpp(chicago, covariates = list(xy_sum = l_im), nd = 20L)
  expect_identical(names(out_im$X), "xy_sum")
  expect_true(all(is.finite(out_im$X$xy_sum)))

  data(simplenet, package = "spatstat.data")
  pts <- spatstat.linnet::runiflpp(8, L = simplenet)
  fac_im <- spatstat.linnet::as.linim(
    spatstat.linnet::linfun(
      function(x, y, seg, tp) factor(ifelse(seg %% 2L == 0L, "even", "odd"), levels = c("odd", "even")),
      L = simplenet
    ),
    delta = 0.1
  )
  out_fac <- prepare_mlpp(pts, covariates = list(road = fac_im), nd = 10L)
  expect_true(is.factor(out_fac$X$road) || is.character(out_fac$X$road))
  expect_false(anyNA(out_fac$X$road))

  expect_error(prepare_mlpp(chicago, covariates = unname(list(lf)), nd = 20L), "named")
  expect_error(prepare_mlpp(chicago, covariates = list(a = lf, a = lf), nd = 20L), "unique")
  expect_error(prepare_mlpp(chicago, covariates = list(a = 1:5), nd = 20L), "linim")
  expect_error(prepare_mlpp(chicago, nd = 1.5), "positive integer")

  # Overpass network preserves seg/tp identity
  verts <- spatstat.geom::ppp(
    x = c(0.5, 0.5, 0.0, 1.0),
    y = c(0.0, 1.0, 0.5, 0.5),
    window = spatstat.geom::owin(c(0, 1), c(0, 1))
  )
  L_overpass <- spatstat.linnet::linnet(verts, edges = cbind(c(1L, 3L), c(2L, 4L)), sparse = TRUE)
  lp_overpass <- spatstat.linnet::lpp(
    data.frame(x = c(0.5, 0.5, 0.5, 0.2), y = c(0.5, 0.5, 0.8, 0.5), seg = c(1L, 2L, 1L, 2L), tp = c(0.5, 0.5, 0.8, 0.2)),
    L = L_overpass
  )
  seg_cov <- spatstat.linnet::linfun(function(x, y, seg, tp) as.numeric(seg * 100), L = L_overpass)
  dat_ov <- prepare_mlpp(lp_overpass, covariates = list(level = seg_cov), nd = 5L)
  expect_equal(dat_ov$X$level[1], 100)
  expect_equal(dat_ov$X$level[2], 200)
})

test_that("prepare_mlpp converts pre-built planar and network quad schemes", {
  skip_if_not_installed("spatstat.random")
  skip_if_not_installed("spatstat.data")

  W <- spatstat.geom::square(1)
  pp <- spatstat.random::runifpoint(10, win = W)

  Q_bt <- spatstat.geom::quadscheme(pp, nd = 8)
  X_bt <- data.frame(x1 = seq_along(spatstat.geom::is.data(Q_bt)))
  dat_bt <- prepare_mlpp(Q_bt, X = X_bt, objective = "poisson")
  expect_s3_class(dat_bt, "mlpp_data")
  expect_identical(dat_bt$type, "planar")
  expect_true(inherits(dat_bt$window, "owin"))
  expect_null(dat_bt$dummy_intensity)

  Q_log <- spatstat.geom::quadscheme.logi(pp, dummytype = "grid", nd = 8)
  X_log <- data.frame(x1 = seq_along(spatstat.geom::is.data(Q_log)))
  dat_log <- prepare_mlpp(Q_log, X_log, objective = "logistic", interaction_integral = 0.1)
  expect_equal(dat_log$dummy_intensity, rep(Q_log$param$rho, length(dat_log$y)))
  expect_equal(dat_log$interaction_integral, rep(0.1, length(dat_log$y)))

  data(simplenet, package = "spatstat.data")
  lp <- spatstat.linnet::runiflpp(12, L = simplenet)
  Ql <- spatstat.linnet::linequad(lp, nd = 8)
  Ul <- attr(Ql, "situ")
  dat_ql <- prepare_mlpp(Ql, X = data.frame(cx = Ul$data$x), objective = "poisson")
  expect_identical(dat_ql$type, "network")
  expect_true(inherits(dat_ql$window, "linnet"))
})
