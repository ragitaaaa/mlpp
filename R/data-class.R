#' Construct an mlpp Point Process Data Object
#'
#' Creates and validates an S3 object of class `mlpp_data` encapsulating
#' observed points, quadrature background dummy points, covariates, weights,
#' and domain metadata.
#'
#' @param X Covariate data (matrix or data frame).
#' @param y Event/dummy indicator, where `1` denotes observed event and `0` denotes a dummy point.
#' @param locations Spatial locations corresponding to the rows of `X`.
#' @param quadrature_weight Optional numeric vector of Berman-Turner quadrature
#'   weights (\eqn{v_i}). Default is `NULL`.
#' @param dummy_intensity Optional numeric vector of dummy-process intensities
#'   (\eqn{\delta_i}) for logistic objectives. Default is `NULL`.
#' @param statistical_weight Optional numeric vector of spatial/dependence weights
#'   (\eqn{w_i}) for weighted objectives. Default is `NULL`.
#' @param interaction_integral Optional numeric scalar or vector of spatial
#'   interaction integrals (\eqn{A_i}) for iteratively weighted objectives; a
#'   scalar is recycled across all rows of `X`. Default is `NULL`.
#' @param window Optional observation window (`owin`) or linear network (`linnet`).
#'   Default is `NULL`.
#' @param type Spatial domain type: `"planar"` (default) or `"network"`.
#' @param metadata Named list of additional domain or quadrature metadata. Default
#'   is `list()`.
#' @param validate Logical; if `TRUE` (default), validates structural invariants immediately.
#'
#' @details
#' `mlpp_data()` is the low-level constructor for `"mlpp_data"` objects. Most
#' users will call [prepare_mlpp()], which handles planar (`ppp`), linear-network
#' (`lpp`), and pre-built `spatstat` quadrature schemes (`quad`). Scalar
#' `interaction_integral` values are automatically recycled to `nrow(X)`.
#'
#' @return An S3 object of class `"mlpp_data"` containing:
#'   \item{X}{Covariate `data.frame` or numeric `matrix`.}
#'   \item{y}{Binary event/dummy indicator (`1` event, `0` dummy).}
#'   \item{locations}{Spatial locations corresponding to the rows of `X`.}
#'   \item{quadrature_weight}{Quadrature weights \eqn{v_i} (or `NULL`).}
#'   \item{dummy_intensity}{Dummy-process intensities \eqn{\delta_i} (or `NULL`).}
#'   \item{statistical_weight}{Statistical weights \eqn{w_i} (or `NULL`).}
#'   \item{interaction_integral}{Interaction integrals \eqn{A_i} (or `NULL`).}
#'   \item{window}{Observation window (`owin`), network (`linnet`), or `NULL`.}
#'   \item{type}{Domain type (`"planar"` or `"network"`).}
#'   \item{metadata}{List of supplementary metadata.}
#'
#' @references
#' Berman, M., & Turner, T. R. (1992). Approximating point process likelihoods
#' with GLIM. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*,
#' 41(1), 31--38. \doi{10.2307/2347614}
#'
#' Baddeley, A., Coeurjolly, J.-F., Rubak, E., & Waagepetersen, R. (2014).
#' Logistic regression for spatial Gibbs point processes. *Biometrika*,
#' 101(2), 377--392. \doi{10.1093/biomet/ast056}
#'
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' @seealso [prepare_mlpp()], [attach_weights()], [attach_interaction()],
#'   [print.mlpp_data()]
#'
#' @examples
#' dat <- mlpp_data(
#'   X = data.frame(x = c(0.2, 0.8, 0.5)),
#'   y = c(1, 0, 0),
#'   locations = data.frame(x = c(0.2, 0.8, 0.5), y = c(0.3, 0.7, 0.5)),
#'   quadrature_weight = c(0.2, 0.4, 0.4),
#'   type = "planar"
#' )
#' print(dat)
#' @export
mlpp_data <- function(
  X,
  y,
  locations,
  quadrature_weight = NULL,
  dummy_intensity = NULL,
  statistical_weight = NULL,
  interaction_integral = NULL,
  window = NULL,
  type = c("planar", "network"),
  metadata = list(),
  validate = TRUE
) {
  type <- match.arg(type)

  if (!is.null(interaction_integral) &&
    is.numeric(interaction_integral) &&
    length(interaction_integral) == 1L &&
    (is.data.frame(X) || is.matrix(X)) &&
    nrow(X) > 0L) {
    interaction_integral <- rep(interaction_integral, nrow(X))
  }

  out <- structure(
    list(
      X = X,
      y = y,
      locations = locations,
      quadrature_weight = quadrature_weight,
      dummy_intensity = dummy_intensity,
      statistical_weight = statistical_weight,
      interaction_integral = interaction_integral,
      window = window,
      type = type,
      metadata = metadata
    ),
    class = "mlpp_data"
  )

  if (isTRUE(validate)) {
    .check_data(out, objective = NULL)
  }

  out
}

#' Print an mlpp Data Object
#'
#' Prints a compact summary of an `mlpp_data` object, including domain type,
#' event and dummy point counts, covariate names, and presence of quadrature,
#' dummy-intensity, statistical, and interaction-integral weights.
#'
#' @param x An `mlpp_data` object.
#' @param ... Additional arguments (currently unused).
#'
#' @details
#' Displays high-level diagnostics so users can verify quadrature balance and
#' weight availability prior to calling [fit_gbpp()].
#'
#' @return Invisibly returns `x`.
#'
#' @seealso [mlpp_data()], [prepare_mlpp()], [attach_weights()]
#'
#' @examples
#' dat <- mlpp_data(
#'   X = data.frame(x = c(0.2, 0.8, 0.5)),
#'   y = c(1, 0, 0),
#'   locations = data.frame(x = c(0.2, 0.8, 0.5), y = c(0.3, 0.7, 0.5)),
#'   quadrature_weight = c(0.2, 0.4, 0.4),
#'   type = "planar"
#' )
#' print(dat)
#' @keywords internal
#' @export
print.mlpp_data <- function(x, ...) {
  n_events <- sum(x$y == 1)
  n_dummy <- sum(x$y == 0)
  n_covariates <- if (is.data.frame(x$X) || is.matrix(x$X)) ncol(x$X) else 0L

  cat("MLPP Point Process Data Object\n")
  cat("------------------------------\n")
  cat("Domain type        :", x$type, "\n")
  cat("Total observations :", length(x$y), "\n")
  cat("  - Events (y = 1) :", n_events, "\n")
  cat("  - Dummy  (y = 0) :", n_dummy, "\n")
  cat("Covariates (p)     :", n_covariates, "\n")
  if (n_covariates > 0L) {
    cat("Covariate names    :", paste(colnames(x$X), collapse = ", "), "\n")
  }
  cat("Quadrature weights :", if (!is.null(x$quadrature_weight)) "Present" else "None", "\n")
  cat("Dummy intensity    :", if (!is.null(x$dummy_intensity)) "Present" else "None", "\n")
  cat("Statistical weights:", if (!is.null(x$statistical_weight)) "Present" else "None", "\n")
  cat("Interaction int.   :", if (!is.null(x$interaction_integral)) "Present" else "None", "\n")
  invisible(x)
}

#' Attach Statistical Weights to an mlpp Data Object
#'
#' Attaches custom statistical weights \eqn{w_i} (e.g., from pilot intensity
#' estimation or inverse probability weighting) to an existing `mlpp_data` object
#' for use with weighted objectives (`"weighted_poisson"` or `"weighted_logistic"`).
#'
#' @param data An `mlpp_data` object.
#' @param weights A numeric vector of strictly positive finite weights with length
#'   equal to the total number of observations (`length(data$y)`).
#'
#' @details
#' Validates that `weights` contains strictly positive finite numeric entries
#' matching `length(data$y)` and stores them in `data$statistical_weight`.
#'
#' @return The modified `mlpp_data` object with `statistical_weight` attached.
#'
#' @references
#' Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for
#' inhomogeneous spatial point processes. *Biometrika*, 97(4), 867--880.
#' \doi{10.1093/biomet/asq047}
#'
#' Lu, C., Guan, Y., van Lieshout, M.-C., & Xu, G. (2025). XGBoostPP: Tree-based
#' estimation of point process intensity functions. *Journal of Computational and
#' Graphical Statistics*, 35(1), 229--240. \doi{10.1080/10618600.2025.2520582}
#'
#' @seealso [mlpp_data()], [attach_interaction()], [fit_gbpp()]
#'
#' @examples
#' dat <- mlpp_data(
#'   X = data.frame(x = c(0.2, 0.8, 0.5)),
#'   y = c(1, 0, 0),
#'   locations = data.frame(x = c(0.2, 0.8, 0.5), y = c(0.3, 0.7, 0.5)),
#'   quadrature_weight = c(0.2, 0.4, 0.4),
#'   type = "planar"
#' )
#' dat <- attach_weights(dat, c(1.0, 0.5, 0.5))
#' print(dat)
#' @export
attach_weights <- function(data, weights) {
  if (!inherits(data, "mlpp_data")) {
    stop("`data` must be an `mlpp_data` object.", call. = FALSE)
  }

  if (!is.numeric(weights) || length(weights) != length(data$y)) {
    stop(
      "`weights` must be a numeric vector with length equal to the number of quadrature points.",
      call. = FALSE
    )
  }

  if (any(!is.finite(weights)) || any(weights <= 0)) {
    stop(
      "`weights` must contain only strictly positive finite values.",
      call. = FALSE
    )
  }

  data$statistical_weight <- weights
  data
}

#' Validate an mlpp data object
#'
#' @param data An `mlpp_data` object to validate.
#' @param objective Optional objective used to determine required data components.
#' @return Invisibly returns `TRUE` if valid.
#' @keywords internal
#' @noRd
.check_data <- function(data, objective = NULL) {
  if (!inherits(data, "mlpp_data")) {
    stop("`data` must be an `mlpp_data` object.", call. = FALSE)
  }

  if (!is.data.frame(data$X) && !is.matrix(data$X)) {
    stop("`data$X` must be a data frame or matrix.", call. = FALSE)
  }

  if (nrow(data$X) == 0L) {
    stop("`data$X` must contain at least one observation.", call. = FALSE)
  }

  if (!is.numeric(data$y) || length(data$y) != nrow(data$X) || any(!is.finite(data$y))) {
    stop(
      "`data$y` must be finite numeric values with the same number of rows as `data$X`.",
      call. = FALSE
    )
  }

  if (!all(data$y %in% c(0, 1))) {
    stop(
      "`data$y` must contain only 0 (dummy) and 1 (event).",
      call. = FALSE
    )
  }

  n_locations <- if (inherits(data$locations, c("ppp", "lpp"))) {
    spatstat.geom::npoints(data$locations)
  } else if (is.data.frame(data$locations) || is.matrix(data$locations)) {
    nrow(data$locations)
  } else {
    length(data$locations)
  }

  if (n_locations != nrow(data$X)) {
    stop(
      "`data$locations` must have the same number of points as `data$X`.",
      call. = FALSE
    )
  }

  if (!is.null(data$quadrature_weight)) {
    if (!is.numeric(data$quadrature_weight) || length(data$quadrature_weight) != nrow(data$X)) {
      stop(
        "`quadrature_weight` must be a numeric vector with one value per observation.",
        call. = FALSE
      )
    }
    if (any(!is.finite(data$quadrature_weight)) ||
      any(data$quadrature_weight < 0) ||
      sum(data$quadrature_weight) <= 0) {
      stop(
        "`quadrature_weight` must contain finite non-negative values with ",
        "positive total mass.",
        call. = FALSE
      )
    }
  }

  if (!is.null(data$dummy_intensity)) {
    if (!is.numeric(data$dummy_intensity) || length(data$dummy_intensity) != nrow(data$X)) {
      stop(
        "`dummy_intensity` must be a numeric vector with one value per observation.",
        call. = FALSE
      )
    }
    if (any(!is.finite(data$dummy_intensity)) || any(data$dummy_intensity <= 0)) {
      stop("`dummy_intensity` must contain strictly positive finite values.", call. = FALSE)
    }
  }

  if (!is.null(data$statistical_weight)) {
    if (!is.numeric(data$statistical_weight) || length(data$statistical_weight) != nrow(data$X)) {
      stop(
        "`statistical_weight` must be a numeric vector with one value per observation.",
        call. = FALSE
      )
    }
    if (any(!is.finite(data$statistical_weight)) ||
      any(data$statistical_weight < 0) ||
      sum(data$statistical_weight) <= 0) {
      stop(
        "`statistical_weight` must contain finite non-negative values with ",
        "positive sum.",
        call. = FALSE
      )
    }
  }

  if (!is.null(data$interaction_integral)) {
    if (!is.numeric(data$interaction_integral)) {
      stop(
        "`interaction_integral` must be numeric.",
        call. = FALSE
      )
    }

    if (length(data$interaction_integral) != nrow(data$X)) {
      stop(
        "`interaction_integral` must have one value per observation.",
        call. = FALSE
      )
    }

    if (!all(is.finite(data$interaction_integral))) {
      stop(
        "`interaction_integral` must contain only finite values.",
        call. = FALSE
      )
    }
  }

  if (!is.null(objective)) {
    objective <- match.arg(
      objective,
      c(
        "poisson",
        "weighted_poisson",
        "logistic",
        "weighted_logistic",
        "iterative_poisson",
        "iterative_logistic"
      )
    )

    if (objective %in%
      c("poisson", "weighted_poisson", "iterative_poisson") &&
      is.null(data$quadrature_weight)) {
      stop(
        "`quadrature_weight` is required for Poisson objectives.",
        call. = FALSE
      )
    }

    if (objective %in%
      c("logistic", "weighted_logistic", "iterative_logistic") &&
      is.null(data$dummy_intensity)) {
      stop(
        "`dummy_intensity` is required for logistic objectives.",
        call. = FALSE
      )
    }

    if (objective %in% c("weighted_poisson", "weighted_logistic") &&
      is.null(data$statistical_weight)) {
      stop(
        "`statistical_weight` is required for weighted objectives.",
        call. = FALSE
      )
    }

    if (objective == "iterative_poisson" &&
      is.null(data$interaction_integral)) {
      stop(
        "`interaction_integral` is required for the iterative Poisson objective.",
        call. = FALSE
      )
    }

    if (objective == "iterative_logistic" &&
      is.null(data$interaction_integral)) {
      stop(
        "`interaction_integral` is required for the iterative Logistic objective.",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}
