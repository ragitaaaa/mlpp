test_that(".get_shap and explain_mlpp satisfy exact additivity across all 3 backends", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  set.seed(404)
  n <- 80L
  dat <- mlpp_data(
    X = data.frame(
      a = stats::rnorm(n),
      b = stats::rnorm(n),
      grp = factor(sample(c("A", "B", "C"), n, replace = TRUE), levels = c("A", "B", "C"))
    ),
    y = as.numeric(stats::runif(n) < 0.3),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = stats::runif(n, 0.5, 2.0),
    type = "planar"
  )

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    fit <- fit_gbpp(
      dat,
      engine = eng,
      objective = "poisson",
      n_trees = 20L,
      learning_rate = 0.08,
      max_depth = 3L,
      verbose = FALSE
    )

    ex_q <- explain_mlpp(fit, dat, background = "quadrature")
    expect_s3_class(ex_q, "mlpp_explanation")
    link_q <- predict(fit, ex_q$data, type = "link")
    expect_lt(max(abs(rowSums(ex_q$shap) + ex_q$bias - link_q)), 1e-5, label = eng)

    ex_e <- explain_mlpp(fit, dat, background = "events")
    expect_equal(nrow(ex_e$shap), sum(dat$y == 1))

    ex_a <- explain_mlpp(fit, dat, background = "all")
    expect_equal(nrow(ex_a$shap), nrow(dat$X))
  }

  out <- utils::capture.output(ret <- print(ex_q))
  expect_identical(ret, ex_q)
  expect_true(any(grepl("TreeSHAP Model Explanation", out)))
})

test_that("explain_mlpp preserves subsetted vector-baseline TreeSHAP additivity", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  set.seed(1010)
  n <- 50L
  dat <- mlpp_data(
    X = data.frame(a = stats::rnorm(n), b = stats::rnorm(n)),
    y = c(rep(1, 15), rep(0, 35)),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = rep(0.2, n),
    type = "planar"
  )
  bl_vec <- stats::rnorm(n, mean = -0.8, sd = 0.3)

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    fit <- fit_gbpp(
      dat,
      engine = eng,
      objective = "poisson",
      n_trees = 10L,
      learning_rate = 0.1,
      max_depth = 2L,
      baseline = bl_vec,
      verbose = FALSE
    )

    idx_q <- which(dat$y == 0)
    idx_e <- which(dat$y == 1)

    ex_q <- explain_mlpp(fit, dat, background = "quadrature")
    ex_e <- explain_mlpp(fit, dat, background = "events")

    expected_q <- predict(fit, dat$X[idx_q, , drop = FALSE], type = "raw") + bl_vec[idx_q]
    expected_e <- predict(fit, dat$X[idx_e, , drop = FALSE], type = "raw") + bl_vec[idx_e]

    expect_lt(max(abs(rowSums(ex_q$shap) + ex_q$bias - expected_q)), 1e-5)
    expect_lt(max(abs(rowSums(ex_e$shap) + ex_e$bias - expected_e)), 1e-5)
  }
})

test_that("plot.mlpp_explanation renders importance, numeric dependence, and categorical boxplots", {
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
    class = "mlpp_explanation"
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
