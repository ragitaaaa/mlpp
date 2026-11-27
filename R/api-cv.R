#' Independent Stochastic Thinning Cross-Validation
#'
#' Evaluates spatial point process gradient boosted models using independent
#' stochastic thinning. By independently dividing the observed events into `V`
#' folds and retaining the full spatial quadrature dummy points in all folds,
#' this method produces mathematically exact out-of-sample sub-processes
#' while recomputing fold quadrature weights so total spatial domain measure
#' (\eqn{|W|} or \eqn{|\mathcal{L}|}) is conserved.
#'
#' @param data An `mlpp_data` object.
#' @param engine Character string specifying the engine: `"xgboost"` (default),
#'   `"lightgbm"`, or `"catboost"`.
#' @param params List of engine-specific hyperparameters (including cross-engine
#'   aliases mapped by [fit_gbpp()]). Default is `list()`.
#' @param n_trees Number of boosting trees/iterations passed to [fit_gbpp()].
#'   Default is `100L`.
#' @param folds Number of independent thinning folds (\eqn{V}). Default is `5L`.
#' @param objective Point process objective function: `"poisson"`,
#'   `"weighted_poisson"`, `"logistic"`, `"weighted_logistic"`,
#'   `"iterative_poisson"`, or `"iterative_logistic"`. Default is `"poisson"`.
#' @param seed Optional integer random seed used locally for fold assignment
#'   without mutating the caller's global `.Random.seed`. Default is `NULL`.
#' @param ... Additional arguments forwarded to [fit_gbpp()] (e.g.,
#'   `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`,
#'   `random_seed`, `verbose`, `baseline`).
#'
#' @details
#' Under \eqn{V}-fold stochastic thinning (Cronie et al., 2024), observed events
#' (`y == 1`, requiring at least `folds` events) are randomly partitioned into
#' \eqn{V} balanced folds while retaining all spatial quadrature dummy points
#' (`y == 0`) in every fold. For fold \eqn{v}, the held-out test pattern
#' corresponds to retention probability \eqn{p_{\mathrm{test}} = 1/V} and the
#' training pattern has retention probability \eqn{p_{\mathrm{train}} = (V-1)/V}.
#'
#' Quadrature weights \eqn{v_i} for training and test sub-datasets are recomputed
#' cleanly (preserving total window area or linear network length
#' \eqn{\sum_i v_i}; hence `data$quadrature_weight` is required for held-out
#' evaluation regardless of the training `objective`). The fitted training
#' log-intensity baseline is shifted by
#' \eqn{-\log(p_{\mathrm{train}}) = \log(V / (V - 1))} to recover the full
#' unthinned intensity \eqn{\hat{\lambda}(u)}, and [eval_mlpp()] scales predicted
#' intensity by `p_scale = 1 / V` so held-out test metrics are evaluated at the
#' exact theoretical intensity \eqn{\hat{\lambda}(u) / V} of a \eqn{1/V}-thinned
#' process. The caller's global `.Random.seed` is restored on exit when `seed` is
#' provided.
#'
#' @return An object of class `"mlpp_cv"` containing:
#'   \item{metrics}{A named list with `loglik_mean`, `loglik_sd`, `deviance_mean`,
#'     and `deviance_sd` computed across the `V` held-out thinning folds.}
#'   \item{folds}{A length-`V` list of fold evaluation objects, each containing
#'     `predictions`, `y`, `quadrature_weight`, `weight_sum`, `train_weight_sum`,
#'     `locations`, `window`, `loglik`, and `deviance`.}
#'   \item{type}{Character string indicating domain type (`"planar"` or `"network"`).}
#'
#' @references
#' Cronie, O., Moradi, M., & Biscio, C. A. N. (2024). A cross-validation-based
#' statistical theory for point processes. *Biometrika*, 111(2), 625--641.
#' \doi{10.1093/biomet/asad041}
#'
#' Lu, C., Guan, Y., van Lieshout, M.-C., & Xu, G. (2025). XGBoostPP: Tree-based
#' estimation of point process intensity functions. *Journal of Computational and
#' Graphical Statistics*, 35(1), 229--240. \doi{10.1080/10618600.2025.2520582}
#'
#' @seealso [fit_gbpp()], [plot.mlpp_cv()], [print.mlpp_cv()], [eval_mlpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(30, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   cv_res <- cv_mlpp(dat, engine = "xgboost", n_trees = 10L, folds = 3L, seed = 42L)
#'   print(cv_res)
#' }
#' @export
cv_mlpp <- function(
  data,
  engine = c("xgboost", "lightgbm", "catboost"),
  params = list(),
  n_trees = 100L,
  folds = 5L,
  objective = c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  ),
  seed = NULL,
  ...
) {
  if (!inherits(data, "mlpp_data")) {
    stop("`data` must be an `mlpp_data` object.", call. = FALSE)
  }

  engine <- match.arg(engine)
  objective <- match.arg(objective)
  folds <- as.integer(folds)

  if (folds < 2L) {
    stop("`folds` must be at least 2.", call. = FALSE)
  }

  .check_data(data, objective = objective)
  if (is.null(data$quadrature_weight)) {
    stop("`data$quadrature_weight` is required for `cv_mlpp()` evaluation.", call. = FALSE)
  }

  extra_args <- list(...)
  bl_arg <- if ("baseline" %in% names(extra_args)) extra_args$baseline else NULL
  if (!is.null(bl_arg)) {
    if (!is.numeric(bl_arg) || any(!is.finite(bl_arg))) {
      stop("`baseline` must be a finite numeric scalar or vector.", call. = FALSE)
    }
    if (length(bl_arg) != 1L && length(bl_arg) != length(data$y)) {
      stop(
        "`baseline` must be a scalar or have length ", length(data$y),
        ", got length ", length(bl_arg), ".",
        call. = FALSE
      )
    }
  }

  idx_events <- which(data$y == 1)
  idx_dummy <- which(data$y == 0)

  if (length(idx_events) < folds) {
    stop("Number of events must be at least equal to `folds`.", call. = FALSE)
  }

  if (!is.null(seed)) {
    has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (has_seed) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    }
    on.exit(
      {
        if (has_seed) {
          assign(".Random.seed", old_seed, envir = .GlobalEnv)
        } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
          rm(".Random.seed", envir = .GlobalEnv)
        }
      },
      add = TRUE
    )
    set.seed(seed)
  }

  fold_ids <- sample(rep_len(seq_len(folds), length(idx_events)))
  total_measure <- sum(data$quadrature_weight)

  recompute_fold_weights <- function(sub_events, idx_sub) {
    dummy_ppp <- data$metadata$quadrature$dummy
    if (identical(data$type, "planar") &&
      inherits(dummy_ppp, "ppp") &&
      !is.null(data$window) &&
      requireNamespace("spatstat.geom", quietly = TRUE)) {
      pts_coords <- if (inherits(data$locations, "ppp")) {
        spatstat.geom::coords(data$locations)
      } else {
        as.data.frame(data$locations)
      }
      ev_x <- pts_coords$x[sub_events]
      ev_y <- pts_coords$y[sub_events]
      pp_sub <- spatstat.geom::ppp(x = ev_x, y = ev_y, window = data$window, check = FALSE)
      Q_sub <- spatstat.geom::quadscheme(pp_sub, dummy = dummy_ppp)
      return(as.numeric(spatstat.geom::w.quad(Q_sub)))
    }

    if (identical(data$type, "network") && inherits(data$locations, "lpp")) {
      if (length(idx_sub) == 0L) {
        return(numeric(0L))
      }
      seg_all <- data$locations$data$seg
      seg_mass <- tapply(data$quadrature_weight, seg_all, sum)
      seg_sub_chr <- as.character(seg_all[idx_sub])
      seg_counts <- table(seg_sub_chr)
      w_sub <- as.numeric(seg_mass[seg_sub_chr]) / as.numeric(seg_counts[seg_sub_chr])
      return(w_sub)
    }

    w_raw <- data$quadrature_weight[idx_sub]
    s_raw <- sum(w_raw)
    if (s_raw > 0 && total_measure > 0) {
      w_raw * (total_measure / s_raw)
    } else {
      w_raw
    }
  }

  subset_locations <- function(locs, idx) {
    if (inherits(locs, c("ppp", "lpp"))) {
      locs[idx]
    } else {
      locs[idx, , drop = FALSE]
    }
  }

  res_folds <- vector("list", folds)
  ll_res <- numeric(folds)
  dev_res <- numeric(folds)

  for (k in seq_len(folds)) {
    test_events <- idx_events[fold_ids == k]
    train_events <- idx_events[fold_ids != k]

    idx_train <- c(train_events, idx_dummy)
    idx_test <- c(test_events, idx_dummy)

    w_train <- recompute_fold_weights(train_events, idx_train)
    w_test <- recompute_fold_weights(test_events, idx_test)

    data_train <- mlpp_data(
      X = data$X[idx_train, , drop = FALSE],
      y = data$y[idx_train],
      locations = subset_locations(data$locations, idx_train),
      quadrature_weight = w_train,
      dummy_intensity = data$dummy_intensity[idx_train],
      statistical_weight = data$statistical_weight[idx_train],
      interaction_integral = data$interaction_integral[idx_train],
      window = data$window,
      type = data$type,
      metadata = data$metadata,
      validate = FALSE
    )

    data_test <- mlpp_data(
      X = data$X[idx_test, , drop = FALSE],
      y = data$y[idx_test],
      locations = subset_locations(data$locations, idx_test),
      quadrature_weight = w_test,
      dummy_intensity = data$dummy_intensity[idx_test],
      statistical_weight = data$statistical_weight[idx_test],
      interaction_integral = data$interaction_integral[idx_test],
      window = data$window,
      type = data$type,
      metadata = data$metadata,
      validate = FALSE
    )

    fold_args <- extra_args
    if (!is.null(bl_arg) && length(bl_arg) > 1L) {
      fold_args$baseline <- bl_arg[idx_train]
    }
    model <- do.call(
      fit_gbpp,
      c(
        list(
          data = data_train,
          engine = engine,
          objective = objective,
          params = params,
          n_trees = n_trees
        ),
        fold_args
      )
    )

    p_train <- (folds - 1) / folds
    if (!is.null(bl_arg) && length(bl_arg) > 1L) {
      attr(model, "baseline") <- bl_arg[idx_test] - log(p_train)
    } else {
      bl_cur <- attr(model, "baseline")
      attr(model, "baseline") <- (if (is.null(bl_cur)) 0.0 else bl_cur) - log(p_train)
    }

    pred_full <- predict(model, newdata = data_test$X, type = "intensity")

    p_test <- 1 / folds
    eval_vals <- eval_mlpp(
      pred_full,
      data_test$y,
      data_test$quadrature_weight,
      metric = "all",
      p_scale = p_test
    )
    ll <- unname(eval_vals["loglik"])
    dev <- unname(eval_vals["deviance"])

    ll_res[k] <- ll
    dev_res[k] <- dev

    res_folds[[k]] <- list(
      predictions = pred_full,
      y = data_test$y,
      quadrature_weight = data_test$quadrature_weight,
      weight_sum = sum(data_test$quadrature_weight),
      train_weight_sum = sum(data_train$quadrature_weight),
      locations = data_test$locations,
      window = data_test$window,
      loglik = ll,
      deviance = dev
    )
  }

  metrics <- list(
    loglik_mean = mean(ll_res),
    loglik_sd = sd(ll_res),
    deviance_mean = mean(dev_res),
    deviance_sd = sd(dev_res)
  )

  structure(
    list(
      metrics = metrics,
      folds = res_folds,
      type = data$type
    ),
    class = "mlpp_cv"
  )
}
