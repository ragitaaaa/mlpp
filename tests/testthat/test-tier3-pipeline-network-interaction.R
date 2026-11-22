test_that("network interaction integral handles a constant pair correlation", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  locations <- spatstat.linnet::runiflpp(
    n = 4,
    L = simplenet
  )

  pair_correlation <- function(distance) {
    rep(1.5, length(distance))
  }

  result <- calc_net_interaction(
    locations = locations,
    network = simplenet,
    pair_correlation = pair_correlation,
    exact = TRUE
  )

  unit_function <- spatstat.linnet::linfun(
    function(x, y, seg, tp) {
      rep(1, length(x))
    },
    L = simplenet
  )

  network_length <- spatstat.linnet::integral.linfun(
    unit_function,
    exact = TRUE
  )

  expect_type(result, "double")
  expect_length(result, 4L)
  expect_equal(
    result,
    rep(0.5 * network_length, 4L),
    tolerance = 1e-8
  )
})


test_that("network interaction integral handles distance-dependent pair correlation", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  locations <- spatstat.linnet::runiflpp(
    n = 3,
    L = simplenet
  )

  pair_correlation <- function(distance) {
    1 + 0.5 * exp(-distance)
  }

  result <- calc_net_interaction(
    locations = locations,
    network = simplenet,
    pair_correlation = pair_correlation,
    exact = TRUE
  )

  expect_type(result, "double")
  expect_length(result, 3L)
  expect_true(all(is.finite(result)))
})


test_that("network interaction integral rejects incompatible networks", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  locations <- spatstat.linnet::runiflpp(
    n = 3,
    L = simplenet
  )

  other_network <- spatstat.linnet::as.linnet(
    spatstat.linnet::insertVertices(
      simplenet,
      seg = 1,
      tp = 0.5
    )
  )

  expect_error(
    calc_net_interaction(
      locations = locations,
      network = other_network,
      pair_correlation = function(distance) {
        rep(1, length(distance))
      }
    ),
    "same linear network"
  )
})


test_that("network interaction integral validates its arguments", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  locations <- spatstat.linnet::runiflpp(
    n = 2,
    L = simplenet
  )

  expect_error(
    calc_net_interaction(
      # locations = spatstat.geom::runifpoint(2),
      locations = spatstat.geom::ppp(
        x = c(0.25, 0.75),
        y = c(0.25, 0.75),
        window = spatstat.geom::square(1)
      ),
      network = simplenet,
      pair_correlation = function(distance) {
        distance
      }
    ),
    "must be an `lpp` object"
  )

  expect_error(
    calc_net_interaction(
      locations = locations,
      network = simplenet,
      pair_correlation = 1
    ),
    "must be a function"
  )

  expect_error(
    calc_net_interaction(
      locations = locations,
      network = simplenet,
      pair_correlation = function(distance) {
        distance
      },
      nd = 0
    ),
    "positive integer"
  )
})

test_that("calc_net_interaction supports exact = FALSE and attach_net_interaction validates inputs", {
  skip_if_not_installed("spatstat.data")
  data(simplenet, package = "spatstat.data")

  locations <- spatstat.linnet::runiflpp(n = 4, L = simplenet)
  unit_val <- spatstat.geom::volume(simplenet)
  res_approx <- calc_net_interaction(
    locations = locations,
    network = simplenet,
    pair_correlation = function(d) rep(1.5, length(d)),
    exact = FALSE,
    nd = 20L
  )
  expect_equal(res_approx, rep(0.5 * unit_val, 4L), tolerance = 1e-6)

  dat_net <- prepare_gbpp_network(locations, nd = 5L)
  dat_planar <- make_small_gbpp_data()
  other_net <- spatstat.linnet::as.linnet(
    spatstat.linnet::insertVertices(simplenet, seg = 1, tp = 0.5)
  )
  pts_other <- spatstat.linnet::runiflpp(n = 4, L = other_net)

  expect_error(attach_net_interaction(list(), locations, rmax = 0.1), "`data` must be")
  expect_error(attach_net_interaction(dat_planar, locations, rmax = 0.1), "network")
  expect_error(attach_net_interaction(dat_net, pts_other, rmax = 0.1), "same linear network")

  dat_no_meta <- dat_net
  dat_no_meta$metadata <- NULL
  expect_error(attach_net_interaction(dat_no_meta, locations, rmax = 0.1), "quadrature metadata")
})
