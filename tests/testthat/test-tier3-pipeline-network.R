test_that("network pipelines work end-to-end across Poisson, Logistic, and Iterative objectives", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.linnet")
  skip_if_no_python("xgboost")

  set.seed(123)
  fixture <- make_network_integration_fixture(n = 30L)
  base_data <- prepare_gbpp_network(
    point_pattern = fixture$point_pattern,
    covariates = fixture$covariates,
    nd = 20L
  )

  # Fixed interaction supplied directly
  data_fixed <- base_data
  data_fixed$interaction_integral <- rep(0.01, length(data_fixed$y))

  # Estimated network interaction attached via attach_net_interaction()
  data_est <- attach_net_interaction(
    data = base_data,
    point_pattern = fixture$point_pattern,
    rmax = 0.005,
    nr = 32L
  )
  expect_length(data_est$interaction_integral, nrow(data_est$X))
  expect_true(all(is.finite(data_est$interaction_integral)))

  cases <- list(
    list(obj = "poisson", dat = base_data),
    list(obj = "logistic", dat = base_data),
    list(obj = "iterative_poisson", dat = data_fixed),
    list(obj = "iterative_logistic", dat = data_fixed),
    list(obj = "iterative_poisson", dat = data_est),
    list(obj = "iterative_logistic", dat = data_est)
  )

  for (i in seq_along(cases)) {
    obj_name <- cases[[i]]$obj
    dat <- cases[[i]]$dat

    model <- fit_gbpp(
      data = dat,
      engine = "xgboost",
      objective = obj_name,
      params = list(eta = 0.1, max_depth = 2L),
      n_trees = 5L
    )

    expect_s3_class(model, "gbpp_xgboost")
    predictions <- predict(model, newdata = dat$X, type = "intensity")
    expect_type(predictions, "double")
    expect_length(predictions, nrow(dat$X))
    expect_true(all(is.finite(predictions)) && all(predictions > 0), info = obj_name)
  }
})
