test_that("planar interaction integral estimator computes K(rmax) - pi * rmax^2", {
  skip_if_not_installed("spatstat.explore")

  pp <- spatstat.geom::ppp(
    x = c(0.2, 0.3, 0.7, 0.8, 0.5),
    y = c(0.2, 0.8, 0.3, 0.7, 0.5),
    window = spatstat.geom::square(1)
  )
  lambda <- rep(5, spatstat.geom::npoints(pp))
  rmax <- 0.1

  K <- spatstat.explore::Kinhom(pp, lambda = lambda, r = c(0, rmax), correction = "isotropic")
  K_df <- as.data.frame(K)
  expected <- K_df$iso[which.min(abs(K_df$r - rmax))] - pi * rmax^2

  expect_equal(
    mlpp:::.est_interaction(pp, rmax = rmax, lambda = lambda, correction = "isotropic"),
    expected
  )
  expect_no_error(mlpp:::.est_interaction(pp, rmax = rmax, lambda = lambda, correction = "border"))
  expect_no_error(mlpp:::.est_interaction(pp, rmax = rmax, lambda = lambda, correction = "translate"))
  expect_error(mlpp:::.est_interaction(pp, rmax = 0), "positive finite numeric")
  expect_error(mlpp:::.est_interaction(pp, rmax = 0.1, lambda = c(1, 2)), "one value for each point")
  expect_error(mlpp:::.est_interaction(pp, rmax = 0.1, lambda = c(1, 2, 3, 4, NA)), "positive finite")
})

test_that("network interaction integral helpers integrate over linnet cleanly", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  locations <- spatstat.linnet::runiflpp(n = 4, L = simplenet)
  net_len <- spatstat.geom::volume(simplenet)

  res_exact <- mlpp:::.net_integral(
    locations = locations,
    network = simplenet,
    pair_correlation = function(d) rep(1.5, length(d)),
    exact = TRUE
  )
  expect_equal(res_exact, rep(0.5 * net_len, 4L), tolerance = 1e-8)

  res_approx <- mlpp:::.net_integral(
    locations = locations,
    network = simplenet,
    pair_correlation = function(d) rep(1.5, length(d)),
    exact = FALSE,
    nd = 20L
  )
  expect_equal(res_approx, rep(0.5 * net_len, 4L), tolerance = 1e-6)

  other_net <- spatstat.linnet::as.linnet(spatstat.linnet::insertVertices(simplenet, seg = 1, tp = 0.5))
  expect_error(
    mlpp:::.net_integral(locations, other_net, function(d) rep(1, length(d))),
    "same linear network"
  )

  expect_error(mlpp:::.est_interaction(locations, rmax = 0), "positive finite")
  expect_error(mlpp:::.est_interaction(locations, rmax = 0.1, correction = "isotropic"), "either")
})

test_that("attach_interaction auto-dispatches for planar and network data and defaults point_pattern", {
  skip_if_not_installed("spatstat.explore")
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.linnet")
  data(lansing, package = "spatstat.data")

  # Planar via explicit point_pattern
  n <- 20L
  dat_p <- mlpp_data(
    X = matrix(rnorm(n * 2), ncol = 2),
    y = c(rep(1, 5), rep(0, 15)),
    locations = matrix(runif(n * 2), ncol = 2),
    quadrature_weight = rep(1 / n, n),
    dummy_intensity = rep(1, n),
    type = "planar"
  )
  expect_error(attach_interaction(dat_p, rmax = 0.2), "metadata\\$point_pattern")
  expect_error(attach_interaction(dat_p, rmax = 0.2, point_pattern = lansing), "contains 5 event")

  res_p <- attach_interaction(dat_p, rmax = 0.2, point_pattern = lansing[1:5])
  expect_s3_class(res_p, "mlpp_data")
  expect_length(res_p$interaction_integral, n)
  expect_true(all(res_p$interaction_integral == res_p$interaction_integral[1]))

  # Network via metadata default and locations fallback
  fixture <- make_network_integration_fixture(n = 12L)
  dat_net <- prepare_mlpp(fixture$point_pattern, covariates = fixture$covariates, nd = 20L)
  res_net <- attach_interaction(dat_net, rmax = 0.1, nr = 32L)
  expect_length(res_net$interaction_integral, nrow(dat_net$X))
  expect_true(all(is.finite(res_net$interaction_integral)))

  dat_no_quad <- dat_net
  dat_no_quad$metadata$quadrature <- NULL
  res_fallback <- attach_interaction(dat_no_quad, rmax = 0.1, nr = 16L, exact = FALSE, nd = 20L)
  expect_length(res_fallback$interaction_integral, nrow(dat_net$X))
  expect_true(all(is.finite(res_fallback$interaction_integral)))
})
