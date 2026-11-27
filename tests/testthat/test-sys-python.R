test_that("package metadata is valid", {
  ver <- utils::packageVersion("mlpp")
  expect_true(is.package_version(ver))
})

test_that("mlpp Python backend path resolves cleanly", {
  py_dir <- mlpp:::.py_path()
  expect_type(py_dir, "character")
  expect_true(dir.exists(py_dir))
  expect_true(file.exists(file.path(py_dir, "gbpp_xgboost.py")))
  expect_true(file.exists(file.path(py_dir, "gbpp_lightgbm.py")))
  expect_true(file.exists(file.path(py_dir, "gbpp_catboost.py")))
})

test_that("mlpp Python environment configures and reports all boosting packages", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  expect_no_error(mlpp:::.init_python())
  expect_no_error(reticulate::py_config())
  expect_true(reticulate::py_available())

  expect_no_error(reticulate::import("xgboost"))
  expect_no_error(reticulate::import("lightgbm"))
  expect_no_error(reticulate::import("catboost"))

  out <- utils::capture.output(status <- check_mlpp_python())
  expect_type(status, "logical")
  expect_true(all(c("numpy", "scipy", "pandas", "lightgbm", "catboost", "xgboost") %in% names(status)))
  expect_true(all(status))
  expect_true(any(grepl("MLPP Python Environment Diagnostic", out)))
})
