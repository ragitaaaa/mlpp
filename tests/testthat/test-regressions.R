# ==============================================================================
# End-to-End Regression & Namespace Cleanliness Suite
# ==============================================================================

test_that("XGBoost Poisson conserves spatial integral and agrees with LightGBM logistic", {
  skip_if_not_installed("spatstat.random")
  skip_if_no_python(c("xgboost", "lightgbm"))

  set.seed(101)
  W <- spatstat.geom::owin(c(0, 10), c(0, 10))
  pp <- spatstat.random::rpoispp(function(x, y) 0.4 * exp(0.15 * x - 0.1 * y), win = W)
  covs <- list(
    cx = spatstat.geom::as.im(function(x, y) x, W = W),
    cy = spatstat.geom::as.im(function(x, y) y, W = W)
  )
  dat <- prepare_mlpp(pp, covariates = covs, nd = 12L)

  fit_xgb <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 60L, learning_rate = 0.05, max_depth = 3L)
  expect_equal(attr(fit_xgb, "params")$base_score, 0.0)

  lam_xgb <- predict(fit_xgb, dat$X, type = "intensity")
  rel_err <- abs(sum(dat$quadrature_weight * lam_xgb) - sum(dat$y)) / sum(dat$y)
  expect_lt(rel_err, 0.05)

  dat_log <- prepare_mlpp(pp, covariates = covs, quadrature_method = "logistic", nd = 12L)
  fit_log_xgb <- fit_gbpp(dat_log, engine = "xgboost", objective = "logistic", n_trees = 40L, learning_rate = 0.05, max_depth = 2L)
  fit_log_lgb <- fit_gbpp(dat_log, engine = "lightgbm", objective = "logistic", n_trees = 40L, learning_rate = 0.05, max_depth = 2L)
  expect_gt(stats::cor(predict(fit_log_xgb, dat_log$X), predict(fit_log_lgb, dat_log$X)), 0.85)
})

test_that("CatBoost stateless Ordered boosting, Bernoulli subsample, multi-threading, and allow_writing_files work", {
  skip_if_no_python("catboost")

  set.seed(202)
  n <- 120L
  dat <- mlpp_data(
    X = data.frame(z1 = stats::rnorm(n), z2 = stats::rnorm(n)),
    y = as.numeric(stats::runif(n) < 0.25),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = seq(0.05, 2.5, length.out = n),
    interaction_integral = 0.01,
    type = "planar"
  )

  tmp_dir <- tempfile("cb_cwd_")
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(tmp_dir)

  fit_ord <- fit_gbpp(
    dat,
    engine = "catboost",
    objective = "poisson",
    n_trees = 40L,
    learning_rate = 0.08,
    max_depth = 3L,
    subsample = 0.8,
    params = list(boosting_type = "Ordered", thread_count = 2L),
    verbose = FALSE
  )

  expect_false(dir.exists(file.path(tmp_dir, "catboost_info")))
  expect_equal(attr(fit_ord, "params")$bootstrap_type, "Bernoulli")
  expect_equal(attr(fit_ord, "params")$allow_writing_files, FALSE)

  lam_ord <- predict(fit_ord, dat$X, type = "intensity")
  expect_lt(abs(sum(dat$quadrature_weight * lam_ord) - sum(dat$y)) / sum(dat$y), 0.1)

  expect_error(
    fit_gbpp(dat, engine = "catboost", objective = "iterative_poisson", n_trees = 5L, params = list(boosting_type = "Ordered")),
    "Ordered"
  )
})

test_that("Factor level subsets, unseen levels, and swapped columns are consistent across all 3 engines", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  set.seed(303)
  n <- 90L
  lvls <- c("Arteri", "Kolektor", "Tol")
  road_fac <- factor(sample(lvls, n, replace = TRUE), levels = lvls)
  x1 <- stats::rnorm(n)
  x2 <- stats::runif(n)
  prob <- ifelse(road_fac == "Tol", 0.7, ifelse(road_fac == "Kolektor", 0.3, 0.1))

  dat <- mlpp_data(
    X = data.frame(x1 = x1, road = road_fac, x2 = x2),
    y = as.numeric(stats::runif(n) < prob),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = rep(1.0, n),
    type = "planar"
  )

  full_tol <- data.frame(x1 = 0.1, road = factor("Tol", levels = lvls), x2 = 0.5)
  char_tol <- data.frame(x2 = 0.5, road = "Tol", x1 = 0.1, stringsAsFactors = FALSE)
  unseen_df <- data.frame(x1 = 0.1, road = factor("Setapak", levels = c(lvls, "Setapak")), x2 = 0.5)
  missing_col_df <- data.frame(x1 = 0.1, road = "Tol")

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    fit <- fit_gbpp(dat, engine = eng, objective = "poisson", n_trees = 25L, learning_rate = 0.1, max_depth = 3L)
    expect_equal(predict(fit, char_tol, type = "link"), predict(fit, full_tol, type = "link"), tolerance = 1e-6)
    expect_error(predict(fit, missing_col_df, type = "link"), "Missing required predictor")
    expect_error(predict(fit, unseen_df, type = "link"), "unseen categorical level")
  }
})

test_that("Vector baseline prevents silent recycling in predict, explain_mlpp, and works in cv_mlpp", {
  skip_if_no_python("xgboost")

  set.seed(808)
  n <- 60L
  dat <- mlpp_data(
    X = data.frame(x1 = stats::rnorm(n), x2 = stats::runif(n)),
    y = c(rep(1, 20), rep(0, 40)),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = c(rep(0, 20), rep(1 / 40, 40)),
    type = "planar"
  )
  bl_vec <- stats::rnorm(n, mean = -1.0, sd = 0.2)

  fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 5L, baseline = bl_vec)
  expect_length(predict(fit, dat$X, type = "link"), n)
  expect_length(predict(fit, dat$X[1:5, , drop = FALSE], type = "raw"), 5L)
  expect_error(predict(fit, dat$X[1:5, , drop = FALSE], type = "link"), "cannot be recycled")

  dat_small <- mlpp_data(
    X = dat$X[1:10, , drop = FALSE],
    y = dat$y[1:10],
    locations = dat$locations[1:10, , drop = FALSE],
    quadrature_weight = rep(0.1, 10),
    type = "planar"
  )
  expect_error(explain_mlpp(fit, dat_small, background = "all"), "observation-specific `baseline` vector")

  cv_bl <- cv_mlpp(dat, engine = "xgboost", objective = "poisson", n_trees = 5L, folds = 3L, seed = 11L, baseline = bl_vec)
  expect_s3_class(cv_bl, "mlpp_cv")
  expect_true(is.finite(cv_bl$metrics$loglik_mean))
})

test_that("Models survive saveRDS/readRDS and save_mlpp/load_mlpp across fresh R sessions", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  set.seed(707)
  n <- 50L
  lvls <- c("A", "B", "C")
  dat <- mlpp_data(
    X = data.frame(num1 = stats::rnorm(n), cat1 = factor(sample(lvls, n, replace = TRUE), levels = lvls)),
    y = as.numeric(stats::runif(n) < 0.3),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = stats::runif(n, 0.5, 1.5),
    type = "planar"
  )

  rscript_exe <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
  skip_if_not(file.exists(rscript_exe), "Rscript binary not found.")

  pkg_root <- normalizePath(testthat::test_path("../.."), winslash = "/")
  new_x <- dat$X[1:6, , drop = FALSE]
  rds_x <- tempfile(fileext = ".rds")
  saveRDS(new_x, rds_x)
  on.exit(unlink(rds_x), add = TRUE)

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    fit <- fit_gbpp(dat, engine = eng, objective = "poisson", n_trees = 10L, learning_rate = 0.1, max_depth = 2L)
    pred_orig <- predict(fit, new_x, type = "link")

    f_rds <- tempfile(fileext = ".rds")
    f_bundle <- tempfile(fileext = ".rds")
    f_script <- tempfile(fileext = ".R")
    on.exit(unlink(c(f_rds, f_bundle, f_script)), add = TRUE)

    saveRDS(fit, f_rds)
    save_mlpp(fit, f_bundle)

    use_pkgload <- file.exists(file.path(pkg_root, "DESCRIPTION"))
    load_stmt <- if (use_pkgload) {
      c(sprintf('setwd("%s")', pkg_root), 'pkgload::load_all(quiet = TRUE)')
    } else {
      c(sprintf('.libPaths(c("%s", .libPaths()))', paste(.libPaths(), collapse = '", "')), 'library(mlpp)')
    }

    script_lines <- c(
      load_stmt,
      sprintf('Sys.setenv(MLPP_PYTHON = "%s")', reticulate::py_config()$python),
      sprintf('nx <- readRDS("%s")', normalizePath(rds_x, winslash = "/")),
      sprintf('m1 <- readRDS("%s")', normalizePath(f_rds, winslash = "/")),
      sprintf('m2 <- load_mlpp("%s")', normalizePath(f_bundle, winslash = "/")),
      'p1 <- predict(m1, nx, type = "link")',
      'p2 <- predict(m2, nx, type = "link")',
      's1 <- mlpp:::.get_shap(m1, nx)',
      'cat(sprintf("P1=%s\\nP2=%s\\nSHAP=%s\\n", paste(p1, collapse=","), paste(p2, collapse=","), paste(rowSums(s1$shap)+s1$bias, collapse=",")))'
    )
    writeLines(script_lines, f_script)

    out <- system2(rscript_exe, args = shQuote(f_script), stdout = TRUE, stderr = TRUE)
    status <- attr(out, "status")
    expect_true(is.null(status) || status == 0L, info = paste(out, collapse = "\n"))

    parse_line <- function(prefix) {
      hit <- grep(paste0("^", prefix, "="), out, value = TRUE)
      as.numeric(strsplit(sub(paste0("^", prefix, "="), "", hit[1]), ",")[[1]])
    }
    expect_equal(parse_line("P1"), pred_orig, tolerance = 1e-5, info = eng)
    expect_equal(parse_line("P2"), pred_orig, tolerance = 1e-5, info = eng)
    expect_equal(parse_line("SHAP"), pred_orig, tolerance = 1e-5, info = eng)
  }
})

test_that("Guards prevent covariate name shadowing, disjoint windows, extreme deltas, and 0-covariate fits", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  # Disjoint im windows & mismatched linnets
  W1 <- spatstat.geom::owin(c(0, 1), c(0, 1))
  W2 <- spatstat.geom::owin(c(5, 6), c(5, 6))
  pp <- spatstat.random::runifpoint(15, win = W1)
  im1 <- spatstat.geom::as.im(function(x, y) 10 + 5 * y, W = W1, dimyx = 16)
  im_dis <- spatstat.geom::as.im(function(x, y) y, W = W2)

  # Covariate named 'x' not shadowed by spatial x
  dat_x <- prepare_mlpp(pp, covariates = list(x = im1), nd = 8L)
  fit_x <- fit_gbpp(dat_x, engine = "xgboost", n_trees = 10L)
  pred_surf <- predict(fit_x, newdata = list(x = im1), type = "im", dimyx = 16)
  coords <- spatstat.geom::raster.xy(pred_surf, drop = TRUE)
  expect_equal(
    pred_surf[list(x = coords$x, y = coords$y)],
    predict(fit_x, newdata = data.frame(x = 10 + 5 * coords$y), type = "intensity"),
    tolerance = 1e-5
  )

  dat_ab <- prepare_mlpp(pp, covariates = list(a = im1, b = im1), nd = 6L)
  fit_ab <- fit_gbpp(dat_ab, engine = "xgboost", n_trees = 4L)
  expect_error(predict(fit_ab, newdata = list(a = im1, b = im_dis), type = "im"), "overlap a common spatial window")

  data(simplenet, package = "spatstat.data")
  net2 <- spatstat.geom::rescale(simplenet, 2)
  lp <- spatstat.linnet::runiflpp(12, L = simplenet)
  lf1 <- spatstat.linnet::linfun(function(x, y, seg, tp) x, L = simplenet)
  lf_other <- spatstat.linnet::linfun(function(x, y, seg, tp) y, L = net2)
  dat_n <- prepare_mlpp(lp, covariates = list(a = lf1, b = lf1), nd = 6L)
  fit_n <- fit_gbpp(dat_n, engine = "xgboost", n_trees = 4L)
  expect_error(predict(fit_n, newdata = list(a = lf1, b = lf_other), type = "linim"), "same linear network")

  # Extreme logistic dummy intensities
  dat_ext <- mlpp_data(
    X = data.frame(x1 = stats::rnorm(20)),
    y = c(rep(1, 5), rep(0, 15)),
    locations = data.frame(x = stats::runif(20), y = stats::runif(20)),
    dummy_intensity = c(rep(1e-300, 10), rep(1e+300, 10)),
    type = "planar"
  )
  for (eng in c("xgboost", "lightgbm", "catboost")) {
    fit_ext <- fit_gbpp(dat_ext, engine = eng, objective = "logistic", n_trees = 3L)
    expect_true(all(is.finite(predict(fit_ext, dat_ext$X, type = "link"))))
  }

  # Zero-covariate rejection
  dat0 <- mlpp_data(
    X = data.frame(row.names = 1:4)[, 0, drop = FALSE],
    y = c(1, 0, 1, 0),
    locations = data.frame(x = c(0, 1, 0.2, 0.8), y = c(0, 1, 0.8, 0.2)),
    quadrature_weight = rep(0.25, 4),
    type = "planar"
  )
  for (eng in c("xgboost", "lightgbm", "catboost")) {
    expect_error(fit_gbpp(dat0, engine = eng, objective = "poisson", n_trees = 5L), "at least one covariate column")
  }
})

test_that("Concise 12-function public NAMESPACE holds and weights/interactions recover signal", {
  ns_exports <- getNamespaceExports("mlpp")
  expect_length(ns_exports, 12L)
  expect_setequal(
    ns_exports,
    c(
      "mlpp_data", "prepare_mlpp", "attach_weights", "attach_interaction",
      "fit_gbpp", "eval_mlpp", "cv_mlpp", "explain_mlpp",
      "save_mlpp", "load_mlpp", "install_mlpp_python", "check_mlpp_python"
    )
  )
  expect_false(any(c(
    "prepare_gbpp", "prepare_gbpp_network", "prepare_quad_data",
    "new_gbpp_data", "validate_gbpp_data", "attach_stat_weight",
    "attach_net_interaction", "est_interaction", "est_net_interaction",
    "calc_interaction", "calc_net_interaction", "calc_dummy_intensity",
    "calc_net_dummy_intensity", "extract_dummy_intensity",
    "calc_iter_poisson_weight", "calc_iter_logistic_weight", "calc_objective",
    "calc_poisson_loglik", "calc_poisson_deviance", "cv_gbpp", "explain_gbpp",
    "save_gbpp_model", "load_gbpp_model", "configure_gbpp_python", "get_gbpp_python"
  ) %in% ns_exports))

  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))
  set.seed(909)
  n <- 160L
  x1 <- stats::runif(n, -2, 2)
  p_ev <- stats::plogis(1.5 * x1 - 0.5)
  dat_unw <- mlpp_data(
    X = data.frame(x1 = x1, x2 = stats::rnorm(n)),
    y = as.numeric(stats::runif(n) < p_ev),
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = rep(0.25, n),
    dummy_intensity = rep(4.0, n),
    statistical_weight = rep(1.0, n),
    interaction_integral = rep(0.4, n),
    type = "planar"
  )
  dat_skew <- attach_weights(dat_unw, ifelse(x1 > 0, 10.0, 0.1))

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    fit_p0 <- fit_gbpp(dat_unw, engine = eng, objective = "poisson", n_trees = 25L, learning_rate = 0.08, max_depth = 3L)
    fit_wp <- fit_gbpp(dat_skew, engine = eng, objective = "weighted_poisson", n_trees = 25L, learning_rate = 0.08, max_depth = 3L)
    fit_il <- fit_gbpp(dat_unw, engine = eng, objective = "iterative_logistic", n_trees = 25L, learning_rate = 0.08, max_depth = 3L)

    p0 <- predict(fit_p0, dat_unw$X, type = "link")
    p_wp <- predict(fit_wp, dat_unw$X, type = "link")
    p_il <- predict(fit_il, dat_unw$X, type = "link")

    expect_gt(stats::cor(p_wp, x1), 0.4)
    expect_gt(stats::cor(p_il, x1), 0.4)
    expect_gt(max(abs(p_wp - p0)), 0.05)
  }
})
