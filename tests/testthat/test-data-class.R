test_that("mlpp_data constructs, validates, and prints objects", {
  X <- data.frame(x1 = c(1, 2, 3), x2 = c(4, 5, 6))
  y <- c(1, 0, 0)
  locations <- data.frame(x = c(0.1, 0.2, 0.3), y = c(0.4, 0.5, 0.6))

  data <- mlpp_data(
    X = X,
    y = y,
    locations = locations,
    quadrature_weight = c(0.1, 0.2, 0.3),
    type = "planar",
    interaction_integral = 0.25
  )

  expect_s3_class(data, "mlpp_data")
  expect_equal(nrow(data$X), 3L)
  expect_equal(data$y, c(1, 0, 0))
  expect_equal(data$interaction_integral, rep(0.25, 3L))
  expect_true(mlpp:::.check_data(data, objective = "poisson"))

  out <- utils::capture.output(ret <- print(data))
  expect_identical(ret, data)
  expect_true(any(grepl("MLPP Point Process Data Object", out)))
})

test_that(".check_data enforces invariants and objective requirements", {
  X <- data.frame(x1 = c(1, 2, 3))
  locations <- data.frame(x = c(0.1, 0.2, 0.3), y = c(0.4, 0.5, 0.6))

  expect_error(
    mlpp_data(
      X = X,
      y = c(1, -1, 0),
      locations = locations,
      quadrature_weight = c(0.1, 0.2, 0.3)
    ),
    "only 0"
  )

  bad_empty <- structure(
    list(
      X = data.frame(a = numeric(0)),
      y = numeric(0),
      locations = data.frame(x = numeric(0), y = numeric(0)),
      type = "planar"
    ),
    class = "mlpp_data"
  )
  expect_error(mlpp:::.check_data(bad_empty), "at least one observation")

  bad_w <- structure(
    list(
      X = data.frame(a = c(1, 2)),
      y = c(1, 0),
      locations = data.frame(x = c(0, 1), y = c(0, 1)),
      quadrature_weight = c(-0.1, 1.0),
      type = "planar"
    ),
    class = "mlpp_data"
  )
  expect_error(mlpp:::.check_data(bad_w), "non-negative")

  bad_d <- structure(
    list(
      X = data.frame(a = c(1, 2)),
      y = c(1, 0),
      locations = data.frame(x = c(0, 1), y = c(0, 1)),
      dummy_intensity = c(0, 1),
      type = "planar"
    ),
    class = "mlpp_data"
  )
  expect_error(mlpp:::.check_data(bad_d), "strictly positive")

  dat <- mlpp_data(
    X = data.frame(x1 = c(1, 2), x2 = c(2, 1)),
    y = c(1, 0),
    locations = data.frame(x = c(0, 1), y = c(0, 1)),
    quadrature_weight = c(0.25, 0.25),
    dummy_intensity = c(100, 100)
  )

  expect_error(mlpp:::.check_data(dat, objective = "weighted_poisson"), "statistical_weight")
  expect_error(mlpp:::.check_data(dat, objective = "iterative_poisson"), "`interaction_integral` is required")
  expect_error(mlpp:::.check_data(dat, objective = "iterative_logistic"), "`interaction_integral` is required")

  expect_error(
    mlpp_data(
      X = data.frame(x1 = c(1, 2), x2 = c(2, 1)),
      y = c(1, 0),
      locations = data.frame(x = c(0, 1), y = c(0, 1)),
      interaction_integral = 1:3
    ),
    "`interaction_integral` must have one value per observation"
  )
})

test_that("attach_weights attaches and validates statistical weights", {
  dat <- mlpp_data(
    X = data.frame(x1 = c(1, 2)),
    y = c(1, 0),
    locations = data.frame(x = c(0, 1), y = c(0, 1)),
    quadrature_weight = c(0.5, 0.5)
  )

  expect_error(attach_weights(list(), c(1, 1)), "`data` must be an `mlpp_data` object")
  expect_error(attach_weights(dat, c(1, 1, 1)), "number of quadrature points")
  expect_error(attach_weights(dat, c(-1, 1)), "strictly positive")

  dat_wt <- attach_weights(dat, c(0.8, 1.2))
  expect_equal(dat_wt$statistical_weight, c(0.8, 1.2))
  expect_true(mlpp:::.check_data(dat_wt, objective = "weighted_poisson"))
})
