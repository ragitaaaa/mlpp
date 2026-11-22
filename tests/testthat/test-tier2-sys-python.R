test_that("GBPP Python environment can be configured", {
  skip_if_no_python()

  expect_no_error(
    configure_gbpp_python()
  )

  expect_no_error(
    reticulate::py_config()
  )

  expect_true(
    reticulate::py_available()
  )
})


test_that("GBPP Python environment provides required modelling packages", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  configure_gbpp_python()

  expect_no_error(
    reticulate::import("xgboost")
  )

  expect_no_error(
    reticulate::import("lightgbm")
  )

  expect_no_error(
    reticulate::import("catboost")
  )

  out <- utils::capture.output(status <- check_gbpp_python())
  expect_type(status, "logical")
  expect_true(all(c("numpy", "scipy", "pandas", "lightgbm", "catboost", "xgboost") %in% names(status)))
  expect_true(all(status))
  expect_true(any(grepl("GBPP Python Environment Diagnostic", out)))
})
