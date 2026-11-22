test_that("explain_gbpp public generic works correctly", {
  skip_if_no_python("xgboost")
  
  data <- new_gbpp_data(
    X = data.frame(
      x = c(0.5, 1.5, 2.5, 3.5),
      y = c(0.5, 1.5, 2.5, 3.5),
      v1 = c(1, 3, 5, 7),
      v2 = c(2, 4, 6, 8)
    ),
    y = c(1, 1, 0, 0),
    locations = data.frame(
      x = c(0.5, 1.5, 2.5, 3.5),
      y = c(0.5, 1.5, 2.5, 3.5)
    ),
    quadrature_weight = rep(1, 4)
  )

  model <- fit_gbpp(data, engine = "xgboost", objective = "poisson", params = list(eta = 0.1, max_depth = 2L), n_trees = 2L)
  
  # Test default (quadrature)
  exp_quad <- explain_gbpp(model, data)
  expect_s3_class(exp_quad, "gbpp_explanation")
  expect_equal(nrow(exp_quad$shap), 2)
  expect_equal(nrow(exp_quad$data), 2)
  expect_equal(exp_quad$data$v1, c(5, 7))
  
  # Test events
  exp_events <- explain_gbpp(model, data, background = "events")
  expect_equal(nrow(exp_events$shap), 2)
  expect_equal(nrow(exp_events$data), 2)
  expect_equal(exp_events$data$v1, c(1, 3))
  
  # Test all
  exp_all <- explain_gbpp(model, data, background = "all")
  expect_equal(nrow(exp_all$shap), 4)
  expect_equal(nrow(exp_all$data), 4)
  expect_equal(exp_all$data$v1, c(1, 3, 5, 7))

  out <- utils::capture.output(ret <- print(exp_quad))
  expect_identical(ret, exp_quad)
  expect_true(any(grepl("TreeSHAP Model Explanation", out)))
})

test_that("plot.gbpp_explanation renders importance, numeric dependence, and categorical boxplots", {
  exp_demo <- structure(
    list(
      shap = matrix(
        c(0.2, -0.1, 0.3, -0.2, 0.05, 0.1, 0.05, -0.05, 0.1, -0.1, -0.2, 0.15,
          0.1, 0.2, -0.1, 0.05, -0.2, 0.15, -0.05, 0.05, 0.2),
        ncol = 3,
        dimnames = list(NULL, c("x", "y", "grp"))
      ),
      bias = rep(-1.0, 7),
      data = data.frame(
        x = c(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9),
        y = c(0.9, 0.7, 0.5, 0.3, 0.4, 0.6, 0.2),
        grp = factor(c("A", "B", "A", "C", "B", "C", "A"))
      )
    ),
    class = "gbpp_explanation"
  )

  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit({
    if (grDevices::dev.cur() > 1L) grDevices::dev.off()
    unlink(tmp_pdf)
  }, add = TRUE)

  expect_identical(plot(exp_demo, type = "importance"), exp_demo)
  expect_identical(plot(exp_demo, type = "dependence"), exp_demo)
  expect_identical(plot(exp_demo, type = "dependence", feature = "grp"), exp_demo)
  expect_error(plot(exp_demo, type = "dependence", feature = "missing_col"), "not found")
})
