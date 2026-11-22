test_that("network interaction integral can be estimated from linearpcfinhom", {
  skip_if_not_installed("spatstat.data")
  set.seed(123)

  data(simplenet, package = "spatstat.data")

  point_pattern <- spatstat.linnet::runiflpp(
    n = 30,
    L = simplenet
  )

  result <- est_net_interaction(
    point_pattern = point_pattern,
    rmax = 0.1,
    nr = 32L
  )

  expect_type(result, "double")
  expect_length(
    result,
    spatstat.geom::npoints(point_pattern)
  )

  expect_true(all(is.finite(result)))
})


test_that("network interaction estimator validates rmax", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  point_pattern <- spatstat.linnet::runiflpp(
    n = 20,
    L = simplenet
  )

  expect_error(
    est_net_interaction(
      point_pattern = point_pattern,
      rmax = 0
    ),
    "positive finite"
  )

  expect_error(
    est_net_interaction(
      point_pattern = point_pattern,
      rmax = NA_real_
    ),
    "positive finite"
  )
})


test_that("network interaction estimator validates correction", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  point_pattern <- spatstat.linnet::runiflpp(
    n = 20,
    L = simplenet
  )

  expect_error(
    est_net_interaction(
      point_pattern = point_pattern,
      rmax = 0.1,
      correction = "isotropic"
    ),
    "either"
  )
})
