#' Construct a GBPP data object
#'
#' Creates an S3 object of class `gbpp_data` encapsulating observed points,
#' quadrature background dummy points, covariates, weights, and domain metadata.
#'
#' @param X Covariate data (matrix or data frame).
#' @param y Event/dummy indicator, where 1 denotes observed event and 0 denotes a dummy point.
#' @param locations Spatial locations corresponding to the rows of `X`.
#' @param quadrature_weight Berman-Turner quadrature weights.
#' @param dummy_intensity Dummy-process intensity for logistic objectives.
#' @param statistical_weight Spatial/dependence weights for weighted objectives.
#' @param interaction_integral Spatial interaction integral for iteratively weighted objectives.
#' @param window Observation window (`owin`) or linear network (`linnet`).
#' @param type Spatial domain type: `"planar"` or `"network"`.
#' @param metadata Additional arbitrary metadata.
#'
#' @details
#' `new_gbpp_data()` is the low-level constructor for `"gbpp_data"` objects. Most
#' users will call [prepare_gbpp()] or [prepare_gbpp_network()], or convert an
#' existing `spatstat` quadrature scheme with [prepare_quad_data()]. Objects created
#' with `new_gbpp_data()` can be verified against a specific boosting objective via
#' [validate_gbpp_data()].
#'
#' @return An object of class `gbpp_data`.
#'
#' @seealso [prepare_gbpp()], [prepare_gbpp_network()], [validate_gbpp_data()],
#'   [attach_stat_weight()], [print.gbpp_data()]
#'
#' @examples
#' dat <- new_gbpp_data(
#'   X = data.frame(x = c(0.2, 0.8, 0.5)),
#'   y = c(1, 0, 0),
#'   locations = data.frame(x = c(0.2, 0.8, 0.5), y = c(0.3, 0.7, 0.5)),
#'   quadrature_weight = c(0.2, 0.4, 0.4),
#'   type = "planar"
#' )
#' print(dat)
#' @export
#' @aliases gbpp_data
new_gbpp_data <- function(
    X,
    y,
    locations,
    quadrature_weight = NULL,
    dummy_intensity = NULL,
    statistical_weight = NULL,
    interaction_integral = NULL,
    window = NULL,
    type = c("planar", "network"),
    metadata = list()
) {
  type <- match.arg(type)

  structure(
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
    class = "gbpp_data"
  )
}

#' Print a GBPP data object
#'
#' Prints a compact summary of a `gbpp_data` object, including domain type,
#' event and dummy point counts, covariate names, and presence of quadrature,
#' dummy-intensity, statistical, and interaction-integral weights.
#'
#' @param x A `gbpp_data` object.
#' @param ... Additional arguments (currently unused).
#'
#' @details
#' Displays high-level diagnostics so users can verify quadrature balance and
#' weight availability prior to calling [fit_gbpp()].
#'
#' @return Invisibly returns `x`.
#'
#' @seealso [new_gbpp_data()], [prepare_gbpp()], [validate_gbpp_data()]
#'
#' @examples
#' dat <- new_gbpp_data(
#'   X = data.frame(x = c(0.2, 0.8, 0.5)),
#'   y = c(1, 0, 0),
#'   locations = data.frame(x = c(0.2, 0.8, 0.5), y = c(0.3, 0.7, 0.5)),
#'   quadrature_weight = c(0.2, 0.4, 0.4),
#'   type = "planar"
#' )
#' print(dat)
#' @export
print.gbpp_data <- function(x, ...) {
  n_events <- sum(x$y == 1)
  n_dummy <- sum(x$y == 0)
  n_covariates <- if (is.data.frame(x$X) || is.matrix(x$X)) ncol(x$X) else 0L

  cat("GBPP Point Process Data Object\n")
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

#' Attach statistical weights to a GBPP data object
#'
#' Attaches custom statistical weights (e.g., from pilot intensity estimation
#' or inverse probability weighting) to an existing `gbpp_data` object for use with
#' weighted objectives (`"weighted_poisson"` or `"weighted_logistic"`).
#'
#' @param data A `gbpp_data` object.
#' @param weights A numeric vector of positive weights with length equal to
#'   the total number of observations (`length(data$y)`).
#'
#' @details
#' Validates that `weights` contains positive finite numeric entries matching
#' `length(data$y)` and stores them in `data$statistical_weight`.
#'
#' @return The modified `gbpp_data` object with `statistical_weight` attached.
#'
#' @references
#' Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for
#' inhomogeneous spatial point processes. *Biometrika*, 97(4), 867--880.
#' \doi{10.1093/biomet/asq047}
#'
#' @seealso [new_gbpp_data()], [attach_interaction()], [attach_net_interaction()],
#'   [fit_gbpp()]
#'
#' @examples
#' dat <- new_gbpp_data(
#'   X = data.frame(x = c(0.2, 0.8, 0.5)),
#'   y = c(1, 0, 0),
#'   locations = data.frame(x = c(0.2, 0.8, 0.5), y = c(0.3, 0.7, 0.5)),
#'   quadrature_weight = c(0.2, 0.4, 0.4),
#'   type = "planar"
#' )
#' dat <- attach_stat_weight(dat, c(1.0, 0.5, 0.5))
#' print(dat)
#' @export
attach_stat_weight <- function(data, weights) {
  if (!inherits(data, "gbpp_data")) {
    stop("`data` must be a `gbpp_data` object.", call. = FALSE)
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
