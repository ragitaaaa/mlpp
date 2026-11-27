# ============================================================
# mlpp - Shared Feature Encoding, Baseline, Python & Model Helpers
# ============================================================

#' Extract Categorical Levels from Training Features
#'
#' @keywords internal
#' @noRd
.factor_levels <- function(X) {
  if (!is.data.frame(X) || ncol(X) == 0L) {
    return(NULL)
  }

  out <- list()
  for (col in names(X)) {
    v <- X[[col]]
    if (is.data.frame(v)) {
      v <- v[[1L]]
    }
    if (is.factor(v)) {
      out[[col]] <- levels(v)
    } else if (is.character(v)) {
      out[[col]] <- sort(unique(v[!is.na(v)]))
    }
  }

  if (length(out) == 0L) NULL else out
}

#' Align, Validate, and Encode Features Consistently Across Fit & Predict
#'
#' @keywords internal
#' @noRd
.encode_X <- function(
  X,
  factor_levels = NULL,
  feature_names = NULL,
  engine = c("xgboost", "lightgbm", "catboost"),
  unseen = c("error", "na")
) {
  engine <- match.arg(engine)
  unseen <- match.arg(unseen)

  if (!is.data.frame(X) && !is.matrix(X)) {
    stop("`X` must be a numeric matrix or data.frame.", call. = FALSE)
  }

  was_matrix <- is.matrix(X)

  if (!is.null(feature_names) && length(feature_names) > 0L) {
    cols <- colnames(X)
    if (is.null(cols)) {
      if (ncol(X) != length(feature_names)) {
        stop(
          "Matrix `newdata` has ", ncol(X), " column(s), expected ",
          length(feature_names), ".",
          call. = FALSE
        )
      }
      colnames(X) <- feature_names
    } else {
      missing_cols <- setdiff(feature_names, cols)
      if (length(missing_cols) > 0L) {
        stop(
          "Missing required predictor column(s): ",
          paste(missing_cols, collapse = ", "),
          call. = FALSE
        )
      }
      X <- X[, feature_names, drop = FALSE]
    }
  }

  if (was_matrix) {
    if (!is.numeric(X)) {
      stop("Predictor matrix must be numeric.", call. = FALSE)
    }
    if (engine == "catboost") {
      return(as.data.frame(X, check.names = FALSE))
    }
    return(X)
  }

  out <- as.data.frame(X, check.names = FALSE)
  for (col in names(out)) {
    v <- out[[col]]
    if (is.data.frame(v)) {
      v <- v[[1L]]
    }

    has_saved <- !is.null(factor_levels) && (col %in% names(factor_levels))
    is_cat <- is.factor(v) || is.character(v)

    if (has_saved) {
      lvls <- factor_levels[[col]]
      ch <- as.character(v)
      unknown <- !is.na(ch) & !(ch %in% lvls)
      if (any(unknown) && unseen == "error") {
        bad_vals <- paste(unique(ch[unknown]), collapse = ", ")
        stop(
          "Column `", col, "` contains unseen categorical level(s): ",
          bad_vals, ".",
          call. = FALSE
        )
      }
    }

    if (engine %in% c("xgboost", "lightgbm")) {
      if (has_saved) {
        out[[col]] <- as.numeric(factor(ch, levels = lvls))
      } else if (is_cat) {
        lvls <- if (is.factor(v)) levels(v) else sort(unique(v[!is.na(v)]))
        out[[col]] <- as.numeric(factor(as.character(v), levels = lvls))
      } else {
        out[[col]] <- as.numeric(v)
      }
    } else if (is_cat || has_saved) {
      out[[col]] <- as.character(v)
    }
  }

  out
}

#' Initialize and Validate Baseline Log-Intensity Offset
#'
#' Computes the default baseline offset (`log(sum(y) / sum(quadrature_weight))`
#' for Poisson objectives or `0.0` for logistic objectives) when `baseline` is
#' `NULL`, and normalizes scalar or vector baselines across observations.
#'
#' @keywords internal
#' @noRd
.init_baseline <- function(data, objective, baseline = NULL) {
  n <- nrow(data$X)
  if (is.null(baseline)) {
    if (objective %in% c("poisson", "weighted_poisson", "iterative_poisson")) {
      lambda0 <- sum(data$y) / sum(data$quadrature_weight)
      if (lambda0 <= 0 || !is.finite(lambda0)) {
        stop("Initial Poisson intensity must be positive and finite.", call. = FALSE)
      }
      baseline <- log(lambda0)
    } else {
      baseline <- 0.0
    }
  }

  if (!is.numeric(baseline) || any(!is.finite(baseline))) {
    stop("`baseline` must be a finite numeric scalar or vector.", call. = FALSE)
  }
  if (length(baseline) != 1L && length(baseline) != n) {
    stop(
      "`baseline` must be a scalar or have length ", n,
      ", got length ", length(baseline), ".",
      call. = FALSE
    )
  }

  b_num <- as.numeric(baseline)
  vec <- if (length(b_num) == 1L) rep(b_num, n) else b_num
  list(
    vec = vec,
    attr_val = b_num,
    has_nonzero = any(vec != 0.0)
  )
}

#' Attach Standard Metadata and Serialization Payload to a Fitted Model
#'
#' @keywords internal
#' @noRd
.wrap_model <- function(
  raw_model,
  engine,
  objective,
  baseline_attr,
  feature_names,
  factor_levels,
  params,
  n_trees,
  data,
  extra_attrs = list()
) {
  ser <- .pack_model(raw_model, engine)

  attr(raw_model, "engine") <- engine
  attr(raw_model, "objective") <- objective
  attr(raw_model, "baseline") <- baseline_attr
  attr(raw_model, "feature_names") <- feature_names
  attr(raw_model, "factor_levels") <- factor_levels
  attr(raw_model, "params") <- params
  attr(raw_model, "n_trees") <- n_trees
  attr(raw_model, "serialized_model") <- ser$payload
  attr(raw_model, "serialized_format") <- ser$format
  attr(raw_model, "data_info") <- list(
    n_obs = nrow(data$X),
    n_events = sum(data$y == 1),
    type = data$type
  )

  for (nm in names(extra_attrs)) {
    attr(raw_model, nm) <- extra_attrs[[nm]]
  }

  class(raw_model) <- c(paste0("gbpp_", engine), "gbpp_model", "mlpp_model", class(raw_model))
  raw_model
}

#' Validate That All Linear Network Covariates Share the Same `linnet`
#'
#' @keywords internal
#' @noRd
.check_linnet <- function(covariates, network, msg) {
  same_network <- vapply(
    covariates,
    function(z) identical(spatstat.linnet::as.linnet(z), network),
    logical(1)
  )
  if (any(!same_network)) {
    stop(msg, call. = FALSE)
  }
  invisible(TRUE)
}

#' Extract Raw Serialized Bytes / String from a Fitted Python Model
#'
#' @keywords internal
#' @noRd
.pack_model <- function(raw_model, engine) {
  if (engine == "lightgbm") {
    return(list(
      payload = raw_model$model_to_string(),
      format = "lightgbm_txt"
    ))
  }

  ext <- if (engine == "xgboost") ".ubj" else ".cbm"
  tmp <- tempfile(fileext = ext)
  on.exit(unlink(tmp), add = TRUE)
  raw_model$save_model(tmp)
  sz <- file.info(tmp)$size
  list(
    payload = readBin(tmp, what = "raw", n = as.integer(sz)),
    format = if (engine == "xgboost") "xgboost_ubj" else "catboost_cbm"
  )
}

#' Rehydrate a Dead Reticulate External Pointer for an `mlpp_model`
#'
#' @keywords internal
#' @noRd
.unpack_model <- function(object) {
  if (!inherits(object, "python.builtin.object")) {
    return(object)
  }

  if (!reticulate::py_is_null_xptr(object)) {
    return(object)
  }

  payload <- attr(object, "serialized_model")
  engine <- attr(object, "engine")

  if (is.null(payload) || is.null(engine)) {
    stop(
      "The `mlpp_model` Python object pointer is null and cannot be restored ",
      "(missing serialized model payload). Re-fit or use `save_mlpp()`.",
      call. = FALSE
    )
  }

  .init_python()

  restored <- switch(engine,
    xgboost = {
      xgb <- reticulate::import("xgboost", delay_load = FALSE)
      tmp <- tempfile(fileext = ".ubj")
      on.exit(unlink(tmp), add = TRUE)
      writeBin(payload, tmp)
      bst <- xgb$Booster()
      bst$load_model(tmp)
      bst
    },
    lightgbm = {
      lgb <- reticulate::import("lightgbm", delay_load = FALSE)
      lgb$Booster(model_str = as.character(payload))
    },
    catboost = {
      cb <- reticulate::import("catboost", delay_load = FALSE)
      tmp <- tempfile(fileext = ".cbm")
      on.exit(unlink(tmp), add = TRUE)
      writeBin(payload, tmp)
      m <- cb$CatBoostRegressor()
      m$load_model(tmp)
      m
    },
    stop("Unsupported engine for model rehydration: ", engine, call. = FALSE)
  )

  for (nm in c(
    "engine", "objective", "n_trees", "baseline",
    "feature_names", "factor_levels", "cat_features",
    "params", "data_info", "serialized_model", "serialized_format"
  )) {
    attr(restored, nm) <- attr(object, nm)
  }
  class(restored) <- class(object)
  restored
}

#' Save a Fitted mlpp Model to Disk
#'
#' Persists a fitted `mlpp_model` together with its native backend serialization
#' payload and metadata (`baseline`, `feature_names`, `factor_levels`,
#' `objective`, `n_trees`) to an `.rds` file so it can be restored reliably in
#' any fresh R session via [load_mlpp()] or `readRDS()`.
#'
#' @param object A fitted `mlpp_model` object.
#' @param file File path where the `.rds` model bundle will be written.
#'
#' @details
#' Native Python booster pointers stored in external memory cannot survive R
#' serialization on their own. While [fit_gbpp()] already embeds the native
#' serialized payload (`"serialized_model"` attribute) so even plain `saveRDS()` /
#' `readRDS()` restores automatically in [predict.mlpp_model()] and
#' [explain_mlpp()], `save_mlpp()` verifies and refreshes the serialized raw
#' payload (XGBoost UBJ binary, LightGBM model string, or CatBoost `cbm` binary
#' stream) before saving the bundle so [load_mlpp()] can reconstruct a live
#' model reliably in any future R session.
#'
#' @return Invisibly returns `file`.
#'
#' @references
#' Ushey, K., Allaire, J. J., & Tang, Y. (2024). *reticulate: Interface to Python*.
#' R package.
#'
#' @seealso [load_mlpp()], [fit_gbpp()], [predict.mlpp_model()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   tmp <- tempfile(fileext = ".rds")
#'   save_mlpp(fit, tmp)
#'   fit2 <- load_mlpp(tmp)
#'   unlink(tmp)
#' }
#' @export
save_mlpp <- function(object, file) {
  if (!inherits(object, "mlpp_model")) {
    stop("`object` must be an `mlpp_model` object.", call. = FALSE)
  }
  if (!is.character(file) || length(file) != 1L || !nzchar(file)) {
    stop("`file` must be a non-empty file path string.", call. = FALSE)
  }

  live_obj <- .unpack_model(object)
  if (is.null(attr(live_obj, "serialized_model"))) {
    ser <- .pack_model(live_obj, attr(live_obj, "engine"))
    attr(live_obj, "serialized_model") <- ser$payload
    attr(live_obj, "serialized_format") <- ser$format
  }

  saveRDS(live_obj, file = file)
  invisible(file)
}

#' Load a Saved mlpp Model from Disk
#'
#' Loads an `mlpp_model` saved with [save_mlpp()] or `saveRDS()` and immediately
#' rehydrates its Python backend object in the active session.
#'
#' @param file Path to a saved `.rds` model file.
#'
#' @details
#' Reads the `.rds` model bundle created by [save_mlpp()], ensures the Python
#' runtime is initialized, and rehydrates the underlying Python booster from its
#' embedded serialized payload.
#'
#' @return A live `mlpp_model` object ready for [predict.mlpp_model()],
#'   [explain_mlpp()], and [plot.mlpp_model()].
#'
#' @references
#' Ushey, K., Allaire, J. J., & Tang, Y. (2024). *reticulate: Interface to Python*.
#' R package.
#'
#' @seealso [save_mlpp()], [fit_gbpp()], [predict.mlpp_model()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'   reticulate::py_available() &&
#'   reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   tmp <- tempfile(fileext = ".rds")
#'   save_mlpp(fit, tmp)
#'   fit2 <- load_mlpp(tmp)
#'   print(fit2)
#'   unlink(tmp)
#' }
#' @export
load_mlpp <- function(file) {
  if (!is.character(file) || length(file) != 1L || !file.exists(file)) {
    stop("Model file does not exist: ", file, call. = FALSE)
  }
  obj <- readRDS(file)
  .unpack_model(obj)
}

#' Import and Cache the Bundled Python Objective Module for an Engine
#'
#' @keywords internal
#' @noRd
.py_mod <- local({
  cache <- list()

  function(engine = c("xgboost", "lightgbm", "catboost")) {
    engine <- match.arg(engine)
    if (!is.null(cache[[engine]])) {
      return(cache[[engine]])
    }

    .init_python()
    mod_name <- paste0("gbpp_", engine)
    mod <- reticulate::import_from_path(
      module = mod_name,
      path = .py_path(),
      delay_load = FALSE
    )
    cache[[engine]] <<- mod
    mod
  }
})

#' Construct a Python Objective Function or Object for a GBDT Engine
#'
#' @keywords internal
#' @noRd
.make_obj <- function(
  engine = c("xgboost", "lightgbm", "catboost"),
  objective = c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  ),
  v = NULL,
  delta = NULL,
  w = NULL,
  A = NULL
) {
  engine <- match.arg(engine)
  objective <- match.arg(objective)
  .py_mod(engine)$make_objective(
    objective = objective,
    v = v,
    delta = delta,
    w = w,
    A = A
  )
}
