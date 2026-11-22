#' Internal function to map GBPP predictions to a spatstat linim object
#'
#' @param object A `gbpp_model` object.
#' @param newdata A list of `spatstat.linnet::linim` objects or a `linnet` object.
#' @param eps Optional spatial resolution (spacing between sample points).
#' @param dimyx Optional argument accepted for symmetry with planar image
#'   prediction (unused for linear networks).
#' @param ... Additional arguments passed to predict.
#'
#' @return A `spatstat.linnet::linim` object.
#' @keywords internal
#' @noRd
.gbpp_predict_linim <- function(object, newdata, eps = NULL, dimyx = NULL, ...) {
  if (missing(newdata) || is.null(newdata)) {
    stop(
      "`newdata` must be a list of `spatstat.linnet::linim` objects or a ",
      "`linnet` object.",
      call. = FALSE
    )
  }

  if (inherits(newdata, "linnet")) {
    network <- newdata
    covariates <- list()
  } else if (is.list(newdata) && length(newdata) > 0 &&
    all(vapply(newdata, function(x) inherits(x, c("linim", "linfun")), logical(1)))) {
    # Extract network while preserving both linim and linfun objects for exact evaluation
    network <- spatstat.linnet::as.linnet(newdata[[1]])
    .gbpp_check_same_linnet(
      newdata,
      network,
      "All network covariates in `newdata` must be defined on the same linear network."
    )
    covariates <- newdata
  } else {
    stop(
      "`newdata` for type = 'linim' must be a list of `linim` objects, ",
      "`linfun` objects, or a `linnet` object.",
      call. = FALSE
    )
  }

  feature_names <- attr(object, "feature_names")
  if (is.null(feature_names)) feature_names <- object$feature_names

  if (length(feature_names) > 0) {
    required_covs <- setdiff(feature_names, c("x", "y"))
    if (length(required_covs) > 0 && !all(required_covs %in% names(covariates))) {
      missing_covs <- setdiff(required_covs, names(covariates))
      stop(
        sprintf(
          "Missing required covariates in `newdata`: %s",
          paste(missing_covs, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }

  f_predict <- function(x, y, seg, tp) {
    n_pts <- length(x)
    x_num <- as.numeric(x)
    y_num <- as.numeric(y)

    needs_lpp <- any(vapply(
      intersect(feature_names, names(covariates)),
      function(nm) !inherits(covariates[[nm]], "linfun"),
      logical(1)
    ))

    p_lpp <- NULL
    if (needs_lpp) {
      p_lpp <- tryCatch(
        spatstat.linnet::lpp(
          data.frame(x = x_num, y = y_num, seg = as.integer(seg), tp = as.numeric(tp)),
          L = network,
          check = FALSE
        ),
        error = function(e) {
          spatstat.linnet::as.lpp(
            x = x_num, y = y_num, seg = as.integer(seg), tp = as.numeric(tp),
            L = network
          )
        }
      )
    }

    cols_list <- vector("list", length(feature_names))
    names(cols_list) <- feature_names

    for (f in feature_names) {
      if (f %in% names(covariates)) {
        cov_obj <- covariates[[f]]
        v_raw <- if (inherits(cov_obj, "linfun")) {
          cov_obj(x_num, y_num, seg, tp)
        } else {
          cov_obj[p_lpp]
        }
        if (length(v_raw) != n_pts) {
          stop("Network covariate evaluation length mismatch for `", f, "`.", call. = FALSE)
        }
        if (is.factor(v_raw) || is.character(v_raw)) {
          cols_list[[f]] <- as.character(v_raw)
        } else {
          cols_list[[f]] <- as.numeric(v_raw)
        }
      } else if (f == "x") {
        cols_list[[f]] <- x_num
      } else if (f == "y") {
        cols_list[[f]] <- y_num
      }
    }

    df <- as.data.frame(cols_list, stringsAsFactors = FALSE, check.names = FALSE)
    valid <- stats::complete.cases(df)
    out <- rep(NA_real_, n_pts)

    if (any(valid)) {
      df_valid <- df[valid, feature_names, drop = FALSE]
      out[valid] <- predict.gbpp_model(object, newdata = df_valid, type = "intensity", ...)
    }
    out
  }

  lf <- spatstat.linnet::linfun(f_predict, network)
  spatstat.linnet::as.linim(lf, eps = eps)
}
