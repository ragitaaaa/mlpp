# ==============================================================================
# Tier 4: End-to-End Audit Regression Suite
# Covers integral conservation, CatBoost stateless Ordered boosting, factor
# level & schema consistency, CV measure conservation & RNG isolation,
# TreeSHAP baseline additivity, network lpp topology, spatial im/linim mapping,
# strict validation, and cross-session RDS serialization.
# ==============================================================================

test_that("XGBoost Poisson enforces base_score = 0 and conserves spatial integral", {
  skip_if_not_installed("spatstat.random")
  skip_if_no_python("xgboost")

  set.seed(101)
  W <- spatstat.geom::owin(c(0, 10), c(0, 10))
  pp <- spatstat.random::rpoispp(function(x, y) 0.4 * exp(0.15 * x - 0.1 * y), win = W)
  covs <- list(
    cx = spatstat.geom::as.im(function(x, y) x, W = W),
    cy = spatstat.geom::as.im(function(x, y) y, W = W)
  )
  dat <- prepare_gbpp(pp, covariates = covs, nd = 12L)

  fit_xgb <- fit_gbpp(
    dat,
    engine = "xgboost",
    objective = "poisson",
    n_trees = 60L,
    learning_rate = 0.05,
    max_depth = 3L,
    verbose = FALSE
  )

  expect_equal(attr(fit_xgb, "params")$base_score, 0.0)

  lam_xgb <- predict(fit_xgb, dat$X, type = "intensity")
  rel_err <- abs(sum(dat$quadrature_weight * lam_xgb) - sum(dat$y)) / sum(dat$y)
  expect_lt(rel_err, 0.05)

  # Logistic parity between XGBoost and LightGBM
  skip_if_not(reticulate::py_module_available("lightgbm"), "LightGBM Python package is not available.")
  dat_log <- prepare_gbpp(pp, covariates = covs, quadrature_method = "logistic", nd = 12L)
  fit_log_xgb <- fit_gbpp(dat_log, engine = "xgboost", objective = "logistic", n_trees = 40L, learning_rate = 0.05, max_depth = 2L, verbose = FALSE)
  fit_log_lgb <- fit_gbpp(dat_log, engine = "lightgbm", objective = "logistic", n_trees = 40L, learning_rate = 0.05, max_depth = 2L, verbose = FALSE)

  p_xgb <- predict(fit_log_xgb, dat_log$X, type = "link")
  p_lgb <- predict(fit_log_lgb, dat_log$X, type = "link")
  expect_gt(stats::cor(p_xgb, p_lgb), 0.85)
})

test_that("CatBoost stateless Ordered boosting, Bernoulli subsample, multi-threading, and no disk litter work cleanly", {
  skip_if_no_python("catboost")

  set.seed(202)
  n <- 120L
  X <- data.frame(z1 = stats::rnorm(n), z2 = stats::rnorm(n))
  # Deliberately non-uniform quadrature weights sorted strongly by magnitude
  v <- seq(0.05, 2.5, length.out = n)
  y <- as.numeric(stats::runif(n) < 0.25)

  dat <- new_gbpp_data(
    X = X,
    y = y,
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = v,
    dummy_intensity = rep(2.0, n),
    statistical_weight = stats::runif(n, 0.5, 1.5),
    interaction_integral = rep(0.01, n),
    type = "planar"
  )

  # Check allow_writing_files = FALSE avoids creating catboost_info/
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
  rel_err <- abs(sum(dat$quadrature_weight * lam_ord) - sum(dat$y)) / sum(dat$y)
  expect_lt(rel_err, 0.1)

  # Iterative objectives in CatBoost must reject boosting_type = 'Ordered' cleanly
  expect_error(
    fit_gbpp(
      dat,
      engine = "catboost",
      objective = "iterative_poisson",
      n_trees = 5L,
      params = list(boosting_type = "Ordered")
    ),
    "Ordered"
  )
})

test_that("Factor level subsets, unseen levels, and swapped columns are consistent across all 3 engines", {
  skip_if_no_python()

  set.seed(303)
  n <- 90L
  lvls <- c("Arteri", "Kolektor", "Tol")
  road_fac <- factor(sample(lvls, n, replace = TRUE), levels = lvls)
  x1 <- stats::rnorm(n)
  x2 <- stats::runif(n)
  # Signal strongly linked to "Tol" (code 3) vs "Arteri" (code 1)
  prob <- ifelse(road_fac == "Tol", 0.7, ifelse(road_fac == "Kolektor", 0.3, 0.1))
  y <- as.numeric(stats::runif(n) < prob)

  dat <- new_gbpp_data(
    X = data.frame(x1 = x1, road = road_fac, x2 = x2),
    y = y,
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = rep(1.0, n),
    type = "planar"
  )

  full_tol <- data.frame(x1 = 0.1, road = factor("Tol", levels = lvls), x2 = 0.5)
  char_tol <- data.frame(x2 = 0.5, road = "Tol", x1 = 0.1, stringsAsFactors = FALSE)
  unseen_df <- data.frame(x1 = 0.1, road = factor("Setapak", levels = c(lvls, "Setapak")), x2 = 0.5)
  missing_col_df <- data.frame(x1 = 0.1, road = "Tol")

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    if (!reticulate::py_module_available(eng)) next

    fit <- fit_gbpp(
      dat,
      engine = eng,
      objective = "poisson",
      n_trees = 25L,
      learning_rate = 0.1,
      max_depth = 3L,
      verbose = FALSE
    )

    p_ref <- predict(fit, full_tol, type = "link")
    p_sub <- predict(fit, char_tol, type = "link")
    expect_equal(p_sub, p_ref, tolerance = 1e-6, info = paste("Subset/column reorder:", eng))

    expect_error(
      predict(fit, missing_col_df, type = "link"),
      "Missing required predictor",
      info = paste("Missing column guard:", eng)
    )

    expect_error(
      predict(fit, unseen_df, type = "link"),
      "unseen categorical level",
      info = paste("Unseen category guard:", eng)
    )
  }
})

test_that("explain_gbpp satisfies exact SHAP additivity on link scale across all 3 engines", {
  skip_if_no_python()

  set.seed(404)
  n <- 80L
  dat <- new_gbpp_data(
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
    if (!reticulate::py_module_available(eng)) next

    fit <- fit_gbpp(
      dat,
      engine = eng,
      objective = "poisson",
      n_trees = 20L,
      learning_rate = 0.08,
      max_depth = 3L,
      verbose = FALSE
    )

    ex_q <- explain_gbpp(fit, dat, background = "quadrature")
    link_q <- predict(fit, ex_q$data, type = "link")
    expect_lt(
      max(abs(rowSums(ex_q$shap) + ex_q$bias - link_q)),
      1e-5,
      label = paste("SHAP additivity for", eng)
    )
  }
})

test_that("cv_gbpp conserves spatial domain measure and restores caller .Random.seed", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")
  skip_if_no_python("xgboost")

  # Planar Berman-Turner measure conservation
  set.seed(505)
  W <- spatstat.geom::owin(c(0, 8), c(0, 6))
  pp <- spatstat.random::runifpoint(25, win = W)
  covs <- list(z1 = spatstat.geom::as.im(function(x, y) x + 0.2 * y, W = W))
  dat_p <- prepare_gbpp(pp, covariates = covs, nd = 8L)

  set.seed(999)
  r1 <- stats::runif(1)
  set.seed(999)
  seed_before <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)

  cv_p <- cv_gbpp(dat_p, engine = "xgboost", n_trees = 5L, folds = 3L, seed = 42L, verbose = FALSE)
  seed_after <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  expect_identical(seed_after, seed_before)

  r2 <- stats::runif(1)
  expect_equal(r2, r1)

  for (k in seq_len(3L)) {
    expect_equal(cv_p$folds[[k]]$weight_sum, spatstat.geom::area(W), tolerance = 1e-6)
    expect_equal(cv_p$folds[[k]]$train_weight_sum, spatstat.geom::area(W), tolerance = 1e-6)
  }

  # Network measure conservation with lpp
  data(simplenet, package = "spatstat.data")
  lp <- spatstat.linnet::runiflpp(20, L = simplenet)
  lcov <- list(z1 = spatstat.linnet::linfun(function(x, y, seg, tp) x + seg, L = simplenet))
  dat_net <- prepare_gbpp_network(lp, covariates = lcov, nd = 10L)
  cv_n <- cv_gbpp(dat_net, engine = "xgboost", n_trees = 5L, folds = 3L, seed = 42L, verbose = FALSE)

  net_len <- spatstat.geom::volume(simplenet)
  for (k in seq_len(3L)) {
    expect_equal(cv_n$folds[[k]]$weight_sum, net_len, tolerance = 1e-6)
    expect_equal(cv_n$folds[[k]]$train_weight_sum, net_len, tolerance = 1e-6)
  }
  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit({
    if (grDevices::dev.cur() > 1L) grDevices::dev.off()
    unlink(tmp_pdf)
  }, add = TRUE)
  expect_invisible(plot(cv_n))
})

test_that("Covariates named 'x' are not shadowed in im/linim and linfun works in plot()", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")
  skip_if_no_python("xgboost")

  set.seed(606)
  W <- spatstat.geom::owin(c(0, 4), c(0, 4))
  pp <- spatstat.random::runifpoint(20, win = W)
  # Covariate named "x" deliberately equals 10 + 5 * y (NOT spatial coordinate x!)
  cov_x <- spatstat.geom::as.im(function(x, y) 10 + 5 * y, W = W, dimyx = 16)
  dat <- prepare_gbpp(pp, covariates = list(x = cov_x), nd = 8L)

  fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 15L, verbose = FALSE)
  pred_surface <- predict(fit, newdata = list(x = cov_x), type = "im", dimyx = 16)

  coords <- spatstat.geom::raster.xy(pred_surface, drop = TRUE)
  expected_vals <- predict(fit, newdata = data.frame(x = 10 + 5 * coords$y), type = "intensity")
  actual_vals <- pred_surface[list(x = coords$x, y = coords$y)]
  expect_equal(actual_vals, expected_vals, tolerance = 1e-5)

  # linfun directly in predict(type = "linim") and plot(model, list(cov = linfun))
  data(simplenet, package = "spatstat.data")
  lp <- spatstat.linnet::runiflpp(15, L = simplenet)
  lf <- spatstat.linnet::linfun(function(x, y, seg, tp) {
    ifelse(seg == 1L, NA_real_, x + seg)
  }, L = simplenet)
  # Use non-NA linfun for training, then verify plot and NA-preserving linfun at predict
  lf_full <- spatstat.linnet::linfun(function(x, y, seg, tp) x + seg, L = simplenet)
  dat_l <- prepare_gbpp_network(lp, covariates = list(cov = lf_full), nd = 10L)
  fit_l <- fit_gbpp(dat_l, engine = "xgboost", objective = "poisson", n_trees = 10L, verbose = FALSE)

  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit({
    if (grDevices::dev.cur() > 1L) grDevices::dev.off()
    unlink(tmp_pdf)
  }, add = TRUE)
  expect_invisible(plot(fit_l, newdata = list(cov = lf_full)))
  im_na <- predict(fit_l, newdata = list(cov = lf), type = "linim")
  expect_s3_class(im_na, "linim")
  expect_true(anyNA(im_na$v))
})

test_that("Overpass network preserves segment topology and negative network A(u) is preserved", {
  # Build a two-crossing network: vertical seg (0.5, 0)->(0.5, 1) and horizontal seg (0, 0.5)->(1, 0.5)
  # sharing midpoint (0.5, 0.5) on distinct vertices or crossing without node?
  # Two crossing segments in linnet without an intersection vertex represent an overpass!
  verts <- spatstat.geom::ppp(
    x = c(0.5, 0.5, 0.0, 1.0),
    y = c(0.0, 1.0, 0.5, 0.5),
    window = spatstat.geom::owin(c(0, 1), c(0, 1))
  )
  L_overpass <- spatstat.linnet::linnet(
    verts,
    edges = cbind(c(1L, 3L), c(2L, 4L)),
    sparse = TRUE
  )
  # Put events directly at (0.5, 0.5) on segment 1 AND segment 2
  lp_overpass <- spatstat.linnet::lpp(
    data.frame(
      x = c(0.5, 0.5, 0.5, 0.2),
      y = c(0.5, 0.5, 0.8, 0.5),
      seg = c(1L, 2L, 1L, 2L),
      tp = c(0.5, 0.5, 0.8, 0.2)
    ),
    L = L_overpass
  )
  # Segment-dependent covariate: different values on bridge vs underpass at exact same (0.5, 0.5)!
  seg_cov <- spatstat.linnet::linfun(function(x, y, seg, tp) as.numeric(seg * 100), L = L_overpass)
  dat_ov <- prepare_gbpp_network(lp_overpass, covariates = list(level = seg_cov), nd = 5L)

  expect_s3_class(dat_ov$locations, "lpp")
  expect_equal(dat_ov$X$level[1], 100)
  expect_equal(dat_ov$X$level[2], 200)

  # Negative interaction integral preservation for inhibited network pattern
  # Place spaced points >= 0.4 units apart on L_overpass so r in [0, 0.8] has many pairs at large r and 0 pairs at small r
  lp_reg <- spatstat.linnet::lpp(
    data.frame(
      x = c(0.5, 0.5, 0.5, 0.1, 0.9, 0.3),
      y = c(0.1, 0.5, 0.9, 0.5, 0.5, 0.5),
      seg = c(1L, 1L, 1L, 2L, 2L, 2L),
      tp = c(0.1, 0.5, 0.9, 0.1, 0.9, 0.3)
    ),
    L = L_overpass
  )
  dat_reg <- prepare_gbpp_network(lp_reg, covariates = list(level = seg_cov), nd = 5L)
  dat_att <- attach_net_interaction(dat_reg, point_pattern = lp_reg, rmax = 0.8, exact = TRUE)
  expect_true(any(dat_att$interaction_integral < 0))
})

test_that("Strict validation guards empty data, bad weights, bad metrics, and 0-covariate fits", {
  # Empty observations
  bad_empty <- structure(
    list(X = data.frame(a = numeric(0)), y = numeric(0), locations = data.frame(x = numeric(0), y = numeric(0)), type = "planar"),
    class = "gbpp_data"
  )
  expect_error(validate_gbpp_data(bad_empty), "at least one observation")

  # Negative quadrature weight
  bad_w <- structure(
    list(X = data.frame(a = c(1, 2)), y = c(1, 0), locations = data.frame(x = c(0, 1), y = c(0, 1)), quadrature_weight = c(-0.1, 1.0), type = "planar"),
    class = "gbpp_data"
  )
  expect_error(validate_gbpp_data(bad_w), "non-negative")

  # Non-positive dummy_intensity
  bad_d <- structure(
    list(X = data.frame(a = c(1, 2)), y = c(1, 0), locations = data.frame(x = c(0, 1), y = c(0, 1)), dummy_intensity = c(0, 1), type = "planar"),
    class = "gbpp_data"
  )
  expect_error(validate_gbpp_data(bad_d), "strictly positive")

  # Invalid metric inputs
  expect_error(calc_poisson_loglik(c(1, NA), c(1, 0), c(0.5, 0.5)), "finite")
  expect_error(calc_poisson_deviance(c(1, 2), c(1, 0), c(-1, 1)), "non-negative")
  expect_error(calc_poisson_loglik(c(1, 2), c(1, 0), c(0.5, 0.5), p_scale = 0), "p_scale")

  # Zero-covariate rejection in fit_gbpp
  skip_if_no_python()
  dat0 <- new_gbpp_data(
    X = data.frame(row.names = 1:4)[, 0, drop = FALSE],
    y = c(1, 0, 1, 0),
    locations = data.frame(x = c(0, 1, 0.2, 0.8), y = c(0, 1, 0.8, 0.2)),
    quadrature_weight = rep(0.25, 4),
    type = "planar"
  )
  for (eng in c("xgboost", "lightgbm", "catboost")) {
    if (!reticulate::py_module_available(eng)) next
    expect_error(
      fit_gbpp(dat0, engine = eng, objective = "poisson", n_trees = 5L),
      "at least one covariate column"
    )
  }
})

test_that("Models survive saveRDS/readRDS and save_gbpp_model/load_gbpp_model across fresh R sessions", {
  skip_if_no_python()

  set.seed(707)
  n <- 50L
  lvls <- c("A", "B", "C")
  dat <- new_gbpp_data(
    X = data.frame(
      num1 = stats::rnorm(n),
      cat1 = factor(sample(lvls, n, replace = TRUE), levels = lvls)
    ),
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
    if (!reticulate::py_module_available(eng)) next

    fit <- fit_gbpp(
      dat,
      engine = eng,
      objective = "poisson",
      n_trees = 10L,
      learning_rate = 0.1,
      max_depth = 2L,
      verbose = FALSE
    )
    pred_orig <- predict(fit, new_x, type = "link")
    shap_orig <- explain_gbpp(fit, dat, background = "quadrature")

    f_rds <- tempfile(fileext = ".rds")
    f_bundle <- tempfile(fileext = ".rds")
    f_script <- tempfile(fileext = ".R")
    on.exit(unlink(c(f_rds, f_bundle, f_script)), add = TRUE)

    saveRDS(fit, f_rds)
    save_gbpp_model(fit, f_bundle)

    use_pkgload <- file.exists(file.path(pkg_root, "DESCRIPTION"))
    load_stmt <- if (use_pkgload) {
      c(sprintf('setwd("%s")', pkg_root), 'pkgload::load_all(quiet = TRUE)')
    } else {
      c(sprintf('.libPaths(c("%s", .libPaths()))', paste(.libPaths(), collapse = '", "')), 'library(mlpp)')
    }

    script_lines <- c(
      load_stmt,
      sprintf('Sys.setenv(GBPP_PYTHON = "%s")', reticulate::py_config()$python),
      sprintf('nx <- readRDS("%s")', normalizePath(rds_x, winslash = "/")),
      sprintf('m1 <- readRDS("%s")', normalizePath(f_rds, winslash = "/")),
      sprintf('m2 <- load_gbpp_model("%s")', normalizePath(f_bundle, winslash = "/")),
      'p1 <- predict(m1, nx, type = "link")',
      'p2 <- predict(m2, nx, type = "link")',
      's1 <- mlpp:::.gbpp_shap(m1, nx)',
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
    p1 <- parse_line("P1")
    p2 <- parse_line("P2")
    s1 <- parse_line("SHAP")

    expect_equal(p1, pred_orig, tolerance = 1e-5, info = paste("readRDS predict:", eng))
    expect_equal(p2, pred_orig, tolerance = 1e-5, info = paste("load_gbpp_model predict:", eng))
    expect_equal(s1, pred_orig, tolerance = 1e-5, info = paste("readRDS SHAP:", eng))
  }
})

test_that("Concise canonical API, internal helpers, and clean NAMESPACE exports hold", {
  dat <- new_gbpp_data(
    X = data.frame(a = c(1, 2, 3), b = c(0.5, -0.5, 1.0)),
    y = c(1, 0, 1),
    locations = data.frame(x = c(0.1, 0.5, 0.9), y = c(0.2, 0.4, 0.8)),
    quadrature_weight = c(0.25, 0.5, 0.25),
    dummy_intensity = c(2.0, 2.0, 2.0),
    type = "planar"
  )
  pred <- c(-0.2, 0.1, 0.4)

  expect_true(is.finite(calc_poisson_loglik(pred, dat$y, dat$quadrature_weight)))
  expect_true(is.finite(calc_poisson_deviance(pred, dat$y, dat$quadrature_weight)))

  obj_new <- calc_objective(dat, pred, objective = "poisson")
  expect_identical(mlpp:::.gbpp_deriv_poisson(dat, pred), obj_new)

  w_box <- spatstat.geom::owin(c(0, 1), c(0, 1))
  expect_equal(calc_dummy_intensity(c(1, 0, 0), w_box), 2)
  dat_wt <- attach_stat_weight(dat, c(0.8, 1.0, 1.2))
  expect_equal(dat_wt$statistical_weight, c(0.8, 1.0, 1.2))

  expect_length(calc_iter_poisson_weight(c(0.1, -0.2, 0.0), c(-0.05, 0.0, 0.02)), 3L)
  expect_length(calc_iter_logistic_weight(c(0.1, -0.2, 0.0), c(2.0, 2.0, 2.0), c(-0.05, 0.0, 0.02)), 3L)
  expect_type(get_gbpp_python(), "character")

  if (reticulate::py_available(initialize = TRUE)) {
    xgb_obj <- mlpp:::.gbpp_make_objective("xgboost", "poisson", quadrature_weight = dat$quadrature_weight)
    lgb_obj <- mlpp:::.gbpp_make_objective("lightgbm", "poisson", quadrature_weight = dat$quadrature_weight)
    cat_obj <- mlpp:::.gbpp_make_objective("catboost", "poisson", quadrature_weight = dat$quadrature_weight)
    expect_true(is.function(xgb_obj))
    expect_true(is.function(lgb_obj))
    expect_true(reticulate::py_has_attr(cat_obj, "calc_ders_range"))
  }

  # Namespace cleanliness: only canonical public API is exported (no aliases, legacy shims, or .gbpp_* internals)
  ns_exports <- getNamespaceExports("mlpp")
  expect_length(ns_exports, 29L)
  expect_false(any(startsWith(ns_exports, ".")))
  expect_false(any(c(
    "gbpp_objective", "prepare_pp_data", "metric_poisson_loglik", "metric_poisson_deviance",
    "calculate_planar_dummy_intensity", "calculate_network_dummy_intensity",
    "calculate_interaction_integral", "estimate_interaction_integral", "attach_interaction_integral",
    "calculate_network_interaction_integral", "estimate_network_interaction_integral", "attach_network_interaction_integral",
    "calculate_iterative_poisson_weight", "calculate_iterative_logistic_weight",
    "attach_statistical_weight", "gbpp_python_path",
    "fit_gbpp_xgboost", "fit_gbpp_lightgbm", "fit_gbpp_catboost"
  ) %in% ns_exports))
  expect_true(all(c(
    "calc_objective", "calc_poisson_loglik", "calc_poisson_deviance",
    "calc_dummy_intensity", "calc_net_dummy_intensity",
    "calc_interaction", "est_interaction", "attach_interaction",
    "calc_net_interaction", "est_net_interaction", "attach_net_interaction",
    "calc_iter_poisson_weight", "calc_iter_logistic_weight",
    "attach_stat_weight", "prepare_quad_data", "get_gbpp_python"
  ) %in% ns_exports))
})

test_that("Vector baseline prevents silent recycling in predict, explain_gbpp, and works in cv_gbpp", {
  skip_if_no_python("xgboost")

  set.seed(808)
  n <- 60L
  X <- data.frame(x1 = stats::rnorm(n), x2 = stats::runif(n))
  y <- c(rep(1, 20), rep(0, 40))
  locs <- data.frame(x = stats::runif(n), y = stats::runif(n))
  w <- c(rep(0, 20), rep(1 / 40, 40))
  bl_vec <- stats::rnorm(n, mean = -1.0, sd = 0.2)

  dat <- new_gbpp_data(X = X, y = y, locations = locs, quadrature_weight = w, type = "planar")
  fit <- fit_gbpp(
    dat,
    engine = "xgboost",
    objective = "poisson",
    n_trees = 5L,
    baseline = bl_vec,
    verbose = FALSE
  )

  # Same-length prediction succeeds, while mismatched row counts fail cleanly (raw still succeeds)
  expect_length(predict(fit, dat$X, type = "link"), n)
  expect_length(predict(fit, dat$X[1:5, , drop = FALSE], type = "raw"), 5L)
  expect_error(
    predict(fit, dat$X[1:5, , drop = FALSE], type = "link"),
    "cannot be recycled"
  )
  expect_error(
    predict(fit, dat$X[1:5, , drop = FALSE], type = "intensity"),
    "cannot be recycled"
  )

  # explain_gbpp rejects mismatched data row count for vector baseline
  dat_small <- new_gbpp_data(
    X = X[1:10, , drop = FALSE],
    y = y[1:10],
    locations = locs[1:10, , drop = FALSE],
    quadrature_weight = w[1:10],
    type = "planar"
  )
  expect_error(
    explain_gbpp(fit, dat_small, background = "all"),
    "observation-specific `baseline` vector"
  )

  # cv_gbpp subsets vector baseline cleanly across folds
  cv_bl <- cv_gbpp(
    dat,
    engine = "xgboost",
    objective = "poisson",
    n_trees = 5L,
    folds = 3L,
    seed = 11L,
    baseline = bl_vec,
    verbose = FALSE
  )
  expect_s3_class(cv_bl, "gbpp_cv")
  expect_true(is.finite(cv_bl$metrics$loglik_mean))
})

test_that("prepare_quad_data preserves domain window/quadrature and cv_gbpp fails fast on missing quadrature_weight", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")

  W <- spatstat.geom::square(1)
  pp <- spatstat.random::runifpoint(15, win = W)
  Q <- spatstat.geom::quadscheme(pp, nd = 6)
  U <- spatstat.geom::union.quad(Q)
  dat_q <- prepare_quad_data(Q, X = data.frame(cx = U$x), objective = "poisson", type = "planar")

  expect_true(inherits(dat_q$window, "owin"))
  expect_true(!is.null(dat_q$metadata$quadrature))

  data(simplenet, package = "spatstat.data")
  lp <- spatstat.linnet::runiflpp(12, L = simplenet)
  Ql <- spatstat.linnet::linequad(lp, nd = 8)
  Ul <- attr(Ql, "situ")
  dat_ql <- prepare_quad_data(
    Ql,
    X = data.frame(cx = Ul$data$x),
    objective = "poisson",
    type = "network"
  )
  expect_true(inherits(dat_ql$window, "linnet"))
  expect_true(!is.null(dat_ql$metadata$quadrature))

  # Logistic data without quadrature_weight must fail upfront in cv_gbpp()
  dat_no_w <- new_gbpp_data(
    X = data.frame(cx = U$x),
    y = as.integer(spatstat.geom::is.data(Q)),
    locations = data.frame(x = U$x, y = U$y),
    dummy_intensity = rep(36, length(U$x)),
    type = "planar"
  )
  expect_error(
    cv_gbpp(dat_no_w, engine = "xgboost", objective = "logistic", folds = 3L),
    "quadrature_weight"
  )
})

test_that("predict.gbpp_model guards disjoint im windows, mismatched linnets, and extreme logistic dummy intensities", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")
  skip_if_no_python("xgboost")

  W1 <- spatstat.geom::owin(c(0, 1), c(0, 1))
  W2 <- spatstat.geom::owin(c(5, 6), c(5, 6))
  pp <- spatstat.random::runifpoint(15, win = W1)
  im1 <- spatstat.geom::as.im(function(x, y) x, W = W1)
  im2_same <- spatstat.geom::as.im(function(x, y) y, W = W1)
  im2_disjoint <- spatstat.geom::as.im(function(x, y) y, W = W2)

  dat_p <- prepare_gbpp(pp, covariates = list(a = im1, b = im2_same), nd = 6L)
  fit_p <- fit_gbpp(dat_p, engine = "xgboost", n_trees = 4L, verbose = FALSE)
  expect_error(
    predict(fit_p, newdata = list(a = im1, b = im2_disjoint), type = "im"),
    "overlap a common spatial window"
  )

  data(simplenet, package = "spatstat.data")
  net2 <- spatstat.geom::rescale(simplenet, 2)
  lp <- spatstat.linnet::runiflpp(12, L = simplenet)
  lf1 <- spatstat.linnet::linfun(function(x, y, seg, tp) x, L = simplenet)
  lf2_same <- spatstat.linnet::linfun(function(x, y, seg, tp) y, L = simplenet)
  lf2_other <- spatstat.linnet::linfun(function(x, y, seg, tp) y, L = net2)

  dat_n <- prepare_gbpp_network(lp, covariates = list(a = lf1, b = lf2_same), nd = 6L)
  fit_n <- fit_gbpp(dat_n, engine = "xgboost", n_trees = 4L, verbose = FALSE)
  expect_error(
    predict(fit_n, newdata = list(a = lf1, b = lf2_other), type = "linim"),
    "same linear network"
  )

  # Extreme logistic dummy intensities remain overflow-safe across Python backends
  dat_ext <- new_gbpp_data(
    X = data.frame(x1 = stats::rnorm(20)),
    y = c(rep(1, 5), rep(0, 15)),
    locations = data.frame(x = stats::runif(20), y = stats::runif(20)),
    dummy_intensity = c(rep(1e-300, 10), rep(1e+300, 10)),
    type = "planar"
  )
  for (eng in c("xgboost", "lightgbm", "catboost")) {
    if (!reticulate::py_module_available(eng)) next
    fit_ext <- fit_gbpp(dat_ext, engine = eng, objective = "logistic", n_trees = 3L, verbose = FALSE)
    expect_true(all(is.finite(predict(fit_ext, dat_ext$X, type = "link"))), info = eng)
  }
})

test_that("Weighted and iterative objectives respond meaningfully to weights and learn spatial signal", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  set.seed(909)
  n <- 160L
  x1 <- stats::runif(n, -2, 2)
  x2 <- stats::rnorm(n)
  p_ev <- stats::plogis(1.5 * x1 - 0.5)
  y <- as.numeric(stats::runif(n) < p_ev)
  v <- rep(0.25, n)
  delta <- rep(4.0, n)
  sw_unif <- rep(1.0, n)
  sw_skew <- ifelse(x1 > 0, 10.0, 0.1)
  A_nonneg <- rep(0.4, n)

  dat_unw <- new_gbpp_data(
    X = data.frame(x1 = x1, x2 = x2),
    y = y,
    locations = data.frame(x = stats::runif(n), y = stats::runif(n)),
    quadrature_weight = v,
    dummy_intensity = delta,
    statistical_weight = sw_unif,
    interaction_integral = rep(0, n),
    type = "planar"
  )
  dat_skew <- attach_stat_weight(dat_unw, sw_skew)
  dat_iter <- dat_unw
  dat_iter$interaction_integral <- A_nonneg

  for (eng in c("xgboost", "lightgbm", "catboost")) {
    fit_p0 <- fit_gbpp(dat_unw, engine = eng, objective = "poisson", n_trees = 25L, learning_rate = 0.08, max_depth = 3L, verbose = FALSE)
    fit_wp <- fit_gbpp(dat_skew, engine = eng, objective = "weighted_poisson", n_trees = 25L, learning_rate = 0.08, max_depth = 3L, verbose = FALSE)
    fit_wl <- fit_gbpp(dat_skew, engine = eng, objective = "weighted_logistic", n_trees = 25L, learning_rate = 0.08, max_depth = 3L, verbose = FALSE)
    fit_ip <- fit_gbpp(dat_iter, engine = eng, objective = "iterative_poisson", n_trees = 25L, learning_rate = 0.08, max_depth = 3L, verbose = FALSE)
    fit_il <- fit_gbpp(dat_iter, engine = eng, objective = "iterative_logistic", n_trees = 25L, learning_rate = 0.08, max_depth = 3L, verbose = FALSE)

    p0 <- predict(fit_p0, dat_unw$X, type = "link")
    p_wp <- predict(fit_wp, dat_unw$X, type = "link")
    p_wl <- predict(fit_wl, dat_unw$X, type = "link")
    p_ip <- predict(fit_ip, dat_unw$X, type = "link")
    p_il <- predict(fit_il, dat_unw$X, type = "link")

    # Signal recovery: intensity positively correlates with x1
    expect_gt(stats::cor(p_wp, x1), 0.4, label = paste("weighted_poisson cor:", eng))
    expect_gt(stats::cor(p_wl, x1), 0.4, label = paste("weighted_logistic cor:", eng))
    expect_gt(stats::cor(p_ip, x1), 0.4, label = paste("iterative_poisson cor:", eng))
    expect_gt(stats::cor(p_il, x1), 0.4, label = paste("iterative_logistic cor:", eng))

    # Skewed weights materially change fitted predictions relative to unweighted fit
    expect_gt(max(abs(p_wp - p0)), 0.05, label = paste("weight sensitivity:", eng))
  }
})

test_that("explain_gbpp preserves subsetted vector-baseline TreeSHAP additivity across engines", {
  skip_if_no_python(c("xgboost", "lightgbm", "catboost"))

  set.seed(1010)
  n <- 50L
  dat <- new_gbpp_data(
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

    ex_q <- explain_gbpp(fit, dat, background = "quadrature")
    ex_e <- explain_gbpp(fit, dat, background = "events")

    expected_q <- predict(fit, dat$X[idx_q, , drop = FALSE], type = "raw") + bl_vec[idx_q]
    expected_e <- predict(fit, dat$X[idx_e, , drop = FALSE], type = "raw") + bl_vec[idx_e]

    expect_lt(max(abs(rowSums(ex_q$shap) + ex_q$bias - expected_q)), 1e-5, label = paste("quad SHAP vec bl:", eng))
    expect_lt(max(abs(rowSums(ex_e$shap) + ex_e$bias - expected_e)), 1e-5, label = paste("events SHAP vec bl:", eng))
  }
})

test_that("Categorical spatial rasters (factor im and linim) predict cleanly across LightGBM and CatBoost", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")
  skip_if_no_python(c("lightgbm", "catboost"))

  set.seed(1111)
  W <- spatstat.geom::owin(c(0, 1), c(0, 1))
  pp <- spatstat.random::runifpoint(20, win = W)
  im_num <- spatstat.geom::as.im(function(x, y) x + y, W = W, dimyx = c(12, 12))
  im_fac <- spatstat.geom::as.im(
    function(x, y) factor(ifelse(x < 0.5, "A", "B"), levels = c("A", "B")),
    W = W,
    dimyx = c(12, 12)
  )
  covs_p <- list(num = im_num, zone = im_fac)
  dat_p <- prepare_gbpp(pp, covariates = covs_p, nd = 8L)

  data(simplenet, package = "spatstat.data")
  lp <- spatstat.linnet::runiflpp(16, L = simplenet)
  lin_num <- spatstat.linnet::linfun(function(x, y, seg, tp) x + y, L = simplenet)
  lin_fac <- spatstat.linnet::as.linim(
    spatstat.linnet::linfun(
      function(x, y, seg, tp) factor(ifelse(seg %% 2L == 0L, "even", "odd"), levels = c("odd", "even")),
      L = simplenet
    ),
    delta = 0.1
  )
  covs_n <- list(num = lin_num, kind = lin_fac)
  dat_n <- prepare_gbpp_network(lp, covariates = covs_n, nd = 8L)

  for (eng in c("lightgbm", "catboost")) {
    fit_p <- fit_gbpp(dat_p, engine = eng, objective = "poisson", n_trees = 6L, max_depth = 2L, verbose = FALSE)
    res_im <- predict(fit_p, newdata = covs_p, type = "im", dimyx = c(8, 8))
    expect_s3_class(res_im, "im")
    expect_true(all(is.finite(res_im$v[!is.na(res_im$v)])), info = paste("factor im:", eng))

    fit_n <- fit_gbpp(dat_n, engine = eng, objective = "poisson", n_trees = 6L, max_depth = 2L, verbose = FALSE)
    res_linim <- predict(fit_n, newdata = covs_n, type = "linim")
    expect_s3_class(res_linim, "linim")
    expect_true(all(is.finite(res_linim$v[!is.na(res_linim$v)])), info = paste("factor linim:", eng))
  }
})

test_that("Audit regression fixes: engine param warnings, LightGBM metadata, zero CatBoost weights, scalar interaction, plot.gbpp_cv window=NULL, binary metrics", {
  skip_if_not_installed("spatstat.data")
  skip_if_not_installed("spatstat.random")

  # 1. Binary y check in calc_poisson_loglik() / calc_poisson_deviance()
  expect_error(calc_poisson_loglik(c(1, 2), c(2, 0), c(0.5, 0.5)), "0 \\(dummy\\) and 1 \\(event\\)")
  expect_error(calc_poisson_deviance(c(1, 2), c(0.5, 0), c(0.5, 0.5)), "0 \\(dummy\\) and 1 \\(event\\)")

  # 2. Scalar interaction_integral in prepare_gbpp() and prepare_gbpp_network()
  set.seed(1212)
  W <- spatstat.geom::square(1)
  pp <- spatstat.random::runifpoint(12, win = W)
  cov_x <- spatstat.geom::as.im(function(x, y) x, W = W, dimyx = c(8, 8))
  dat_p <- prepare_gbpp(pp, covariates = list(cx = cov_x), nd = 6L, interaction_integral = 0.25)
  expect_equal(dat_p$interaction_integral, rep(0.25, length(dat_p$y)))

  data(simplenet, package = "spatstat.data")
  lp <- spatstat.linnet::runiflpp(10, L = simplenet)
  lin_x <- spatstat.linnet::linfun(function(x, y, seg, tp) x, L = simplenet)
  dat_n <- prepare_gbpp_network(lp, covariates = list(cx = lin_x), nd = 6L, interaction_integral = -0.1)
  expect_equal(dat_n$interaction_integral, rep(-0.1, length(dat_n$y)))

  # 3. plot.gbpp_cv(..., separate = FALSE) works cleanly when window = NULL
  cv_no_win <- structure(
    list(
      metrics = list(loglik_mean = -1.0, loglik_sd = 0.1, deviance_mean = 2.0, deviance_sd = 0.2),
      folds = list(
        list(y = c(1, 0), locations = data.frame(x = c(0.2, 0.5), y = c(0.3, 0.5)), window = NULL),
        list(y = c(1, 0), locations = data.frame(x = c(0.8, 0.5), y = c(0.7, 0.5)), window = NULL)
      ),
      type = "planar"
    ),
    class = "gbpp_cv"
  )
  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit({
    if (grDevices::dev.cur() > 1L) grDevices::dev.off()
    unlink(tmp_pdf)
  }, add = TRUE)
  expect_invisible(plot(cv_no_win, separate = FALSE))

  # 4. LightGBM plain character params$objective + warning & CatBoost warning + zero quadrature weights
  skip_if_no_python(c("lightgbm", "catboost"))

  expect_warning(
    fit_lgb <- fit_gbpp(
      dat_p,
      engine = "lightgbm",
      objective = "poisson",
      n_trees = 5L,
      max_depth = 2L,
      params = list(objective = "regression"),
      verbose = FALSE
    ),
    "Ignoring user-supplied `params\\$objective`"
  )
  expect_identical(attr(fit_lgb, "params")$objective, "poisson")
  expect_false(inherits(attr(fit_lgb, "params")$objective, "python.builtin.object"))

  dat_zero <- new_gbpp_data(
    X = data.frame(x1 = stats::rnorm(20)),
    y = c(rep(1, 5), rep(0, 15)),
    locations = data.frame(x = stats::runif(20), y = stats::runif(20)),
    quadrature_weight = c(rep(0, 5), rep(1 / 15, 15)),
    type = "planar"
  )
  expect_warning(
    fit_cb <- fit_gbpp(
      dat_zero,
      engine = "catboost",
      objective = "poisson",
      n_trees = 5L,
      max_depth = 2L,
      params = list(loss_function = "RMSE"),
      verbose = FALSE
    ),
    "Ignoring user-supplied `params\\$loss_function`"
  )
  preds_cb <- predict(fit_cb, dat_zero$X, type = "link")
  expect_true(all(is.finite(preds_cb)))
})
