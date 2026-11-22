#' Internal function to map GBPP predictions to a spatstat im object
#'
#' @param object A `gbpp_model` object.
#' @param newdata A list of `spatstat.geom::im` objects or an `owin` object.
#' @param eps Optional spatial resolution (grid spacing).
#' @param dimyx Optional spatial resolution (dimensions).
#' @param ... Additional arguments passed to predict.
#'
#' @return A `spatstat.geom::im` object.
#' @keywords internal
#' @noRd
.gbpp_predict_im <- function(object, newdata, eps = NULL, dimyx = NULL, ...) {
  if (missing(newdata) || is.null(newdata)) {
    stop(
      "`newdata` must be a list of `spatstat.geom::im` objects or an `owin` window.",
      call. = FALSE
    )
  }

  if (inherits(newdata, "owin")) {
    window <- newdata
    covariates <- list()
  } else if (is.list(newdata) &&
    length(newdata) > 0 &&
    all(vapply(newdata, inherits, logical(1), "im"))) {
    window <- spatstat.geom::Window(newdata[[1]])
    covariates <- newdata
    disjoint <- vapply(
      covariates,
      function(z) {
        inter <- spatstat.geom::intersect.owin(window, spatstat.geom::Window(z), fatal = FALSE)
        is.null(inter) || spatstat.geom::is.empty(inter) || spatstat.geom::area(inter) <= 0
      },
      logical(1)
    )
    if (any(disjoint)) {
      stop("All `im` covariates in `newdata` must overlap a common spatial window.", call. = FALSE)
    }
  } else {
    stop(
      "`newdata` for type = 'im' must be a list of `im` objects or an `owin` window.",
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

  mask <- spatstat.geom::as.mask(window, eps = eps, dimyx = dimyx)
  coords <- spatstat.geom::raster.xy(mask, drop = FALSE)
  n_pts <- length(coords$x)

  df <- if (length(feature_names) > 0L) {
    cols_list <- vector("list", length(feature_names))
    names(cols_list) <- feature_names
    for (f in feature_names) {
      if (f %in% names(covariates)) {
        v_extracted <- covariates[[f]][list(x = coords$x, y = coords$y)]
        if (is.factor(v_extracted) || is.character(v_extracted)) {
          cols_list[[f]] <- as.character(v_extracted)
        } else {
          cols_list[[f]] <- as.numeric(v_extracted)
        }
      } else if (f == "x") {
        cols_list[[f]] <- as.numeric(coords$x)
      } else if (f == "y") {
        cols_list[[f]] <- as.numeric(coords$y)
      }
    }
    as.data.frame(cols_list, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    data.frame(x = as.numeric(coords$x), y = as.numeric(coords$y))[seq_len(n_pts), 0, drop = FALSE]
  }

  inside <- as.vector(mask$m)
  valid_covariates <- stats::complete.cases(df)
  valid <- inside & valid_covariates

  out_mat <- matrix(NA_real_, nrow = mask$dim[1], ncol = mask$dim[2])

  if (any(valid)) {
    df_valid <- df[valid, feature_names, drop = FALSE]
    preds <- predict.gbpp_model(object, newdata = df_valid, type = "intensity", ...)
    out_mat[valid] <- preds
  }

  spatstat.geom::im(out_mat, xcol = mask$xcol, yrow = mask$yrow)
}
