test_that("package metadata is valid", {
  ver <- utils::packageVersion("mlpp")
  expect_true(is.package_version(ver))
})
