test_that("Poisson derivatives are calculated correctly", {

  data <- new_gbpp_data(
    X = data.frame(x1 = 1:3),
    y = c(1, 0, 0),
    locations = data.frame(
      x = 1:3,
      y = 1:3
    ),
    quadrature_weight = c(0.2, 0.3, 0.5)
  )

  prediction <- c(
    log(2),
    log(3),
    log(4)
  )

  result <- calc_objective(
    data = data,
    prediction = prediction,
    objective = "poisson"
  )

  expected_gradient <- c(
    0.2 * 2 - 1,
    0.3 * 3 - 0,
    0.5 * 4 - 0
  )

  expected_hessian <- c(
    0.2 * 2,
    0.3 * 3,
    0.5 * 4
  )

  expect_equal(
    result$gradient,
    expected_gradient
  )

  expect_equal(
    result$hessian,
    expected_hessian
  )
})

test_that("Weighted Poisson derivatives are calculated correctly", {
  data <- new_gbpp_data(
    X = data.frame(x1 = 1:3),
    y = c(1, 0, 0),
    locations = data.frame(x = 1:3, y = 1:3),
    quadrature_weight = c(0.2, 0.3, 0.5),
    statistical_weight = c(2.0, 0.5, 1.5)
  )

  prediction <- c(log(2), log(3), log(4))
  result <- calc_objective(data, prediction, objective = "weighted_poisson")

  expected_gradient <- c(2.0 * (0.2 * 2 - 1), 0.5 * (0.3 * 3), 1.5 * (0.5 * 4))
  expected_hessian <- c(2.0 * (0.2 * 2), 0.5 * (0.3 * 3), 1.5 * (0.5 * 4))

  expect_equal(result$gradient, expected_gradient, tolerance = 1e-12)
  expect_equal(result$hessian, expected_hessian, tolerance = 1e-12)
})

test_that("Logistic derivatives are calculated correctly", {
  data <- new_gbpp_data(
    X = data.frame(x1 = 1:3),
    y = c(1, 0, 1),
    locations = data.frame(
      x = 1:3,
      y = 1:3
    ),
    dummy_intensity = c(100, 100, 100)
  )

  prediction <- rep(log(100), 3)

  result <- calc_objective(
    data = data,
    prediction = prediction,
    objective = "logistic"
  )

  expected_pi <- rep(0.5, 3)

  expected_gradient <- expected_pi - data$y

  expected_hessian <- rep(0.25, 3)

  expect_equal(
    result$gradient,
    expected_gradient
  )

  expect_equal(
    result$hessian,
    expected_hessian
  )
})

test_that("Weighted logistic derivatives are calculated correctly", {

  data <- new_gbpp_data(
    X = data.frame(x1 = 1:3),
    y = c(1, 0, 1),
    locations = data.frame(
      x = 1:3,
      y = 1:3
    ),
    dummy_intensity = c(100, 100, 100),
    statistical_weight = c(2, 3, 4)
  )

  prediction <- rep(log(100), 3)

  result <- calc_objective(
    data = data,
    prediction = prediction,
    objective = "weighted_logistic"
  )

  expected_pi <- rep(0.5, 3)

  expected_gradient <- c(
    2 * (0.5 - 1),
    3 * (0.5 - 0),
    4 * (0.5 - 1)
  )

  expected_hessian <- c(
    2 * 0.25,
    3 * 0.25,
    4 * 0.25
  )

  expect_equal(
    result$gradient,
    expected_gradient
  )

  expect_equal(
    result$hessian,
    expected_hessian
  )
})

test_that("Logistic derivatives remain finite for large predictions", {

  data <- new_gbpp_data(
    X = data.frame(x1 = 1:2),
    y = c(1, 0),
    locations = data.frame(
      x = 1:2,
      y = 1:2
    ),
    dummy_intensity = c(100, 100)
  )

  prediction <- c(1000, -1000)

  result <- calc_objective(
    data = data,
    prediction = prediction,
    objective = "logistic"
  )

  expect_true(all(is.finite(result$gradient)))
  expect_true(all(is.finite(result$hessian)))
})

test_that("iterative Poisson objective matches theoretical derivatives", {

  data <- new_gbpp_data(
    X = data.frame(
      x1 = c(1, 2, 3, 4),
      x2 = c(4, 3, 2, 1)
    ),
    y = c(1, 0, 1, 0),
    locations = data.frame(
      x = c(0, 1, 2, 3),
      y = c(0, 1, 2, 3)
    ),
    quadrature_weight = c(
      0.2,
      0.3,
      0.4,
      0.5
    ),
    interaction_integral = c(
      0,
      0.5,
      1,
      2
    )
  )

  prediction <- c(
    0,
    0,
    0,
    0
  )

  result <- calc_objective(
    data = data,
    prediction = prediction,
    objective = "iterative_poisson"
  )

  expected_weight <- c(
    1,
    2 / 3,
    1 / 2,
    1 / 3
  )
  mean_wt <- mean(expected_weight)
  expected_weight <- expected_weight / mean_wt

  expected_gradient <- c(
    -0.8,
    0.2,
    -0.3,
    1 / 6
  ) / mean_wt

  expected_hessian <- c(
    0.2,
    0.2,
    0.2,
    1 / 6
  ) / mean_wt

  expect_equal(
    result$weight,
    expected_weight,
    tolerance = 1e-12
  )

  expect_equal(
    result$gradient,
    expected_gradient,
    tolerance = 1e-12
  )

  expect_equal(
    result$hessian,
    expected_hessian,
    tolerance = 1e-12
  )
})

test_that("iterative Poisson reduces to Poisson when interaction is zero", {
  data <- new_gbpp_data(
    X = data.frame(x1 = c(1, 2, 3), x2 = c(3, 2, 1)),
    y = c(1, 0, 1),
    locations = data.frame(x = c(0, 1, 2), y = c(0, 1, 2)),
    quadrature_weight = c(0.2, 0.3, 0.4),
    interaction_integral = c(0, 0, 0)
  )

  prediction <- c(-0.5, 0.2, 0.7)
  poisson <- calc_objective(data, prediction, objective = "poisson")
  iterative <- calc_objective(data, prediction, objective = "iterative_poisson")

  expect_equal(iterative$weight, rep(1, 3))
  expect_equal(iterative$gradient, poisson$gradient)
  expect_equal(iterative$hessian, poisson$hessian)
})

test_that(
  "iterative Logistic objective matches theoretical derivatives",
  {

    data <- new_gbpp_data(
      X = data.frame(
        x1 = c(1, 2, 3, 4),
        x2 = c(4, 3, 2, 1)
      ),
      y = c(1, 0, 1, 0),
      locations = data.frame(
        x = c(0, 1, 2, 3),
        y = c(0, 1, 2, 3)
      ),
      dummy_intensity = c(
        100,
        100,
        100,
        100
      ),
      interaction_integral = c(
        0,
        0.5,
        1,
        2
      )
    )

    prediction <- c(
      -0.5,
      0,
      0.5,
      1
    )

    result <- calc_objective(
      data = data,
      prediction = prediction,
      objective = "iterative_logistic"
    )

    phi <- pmin(
      pmax(prediction, -50),
      50
    )

    lambda <- exp(phi)

    delta <- data$dummy_intensity

    expected_weight <- (
      lambda + delta
    ) / (
      delta *
        (1 + lambda * data$interaction_integral)
    )
    expected_weight <- expected_weight / mean(expected_weight)

    pi <- stats::plogis(
      phi - log(delta)
    )

    expected_gradient <- (
      expected_weight *
        (pi - data$y)
    )

    expected_hessian <- (
      expected_weight *
        pi *
        (1 - pi)
    )

    expect_equal(
      result$weight,
      expected_weight,
      tolerance = 1e-12
    )

    expect_equal(
      result$gradient,
      expected_gradient,
      tolerance = 1e-12
    )

    expect_equal(
      result$hessian,
      expected_hessian,
      tolerance = 1e-12
    )
  }
)
