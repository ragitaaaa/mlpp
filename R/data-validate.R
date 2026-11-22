#' Validate a GBPP data object
#'
#' Checks the structure and required components of a `gbpp_data` object.
#' When an objective is supplied, also checks that the data object contains
#' the quantities required by that objective.
#'
#' @param data A `gbpp_data` object to validate.
#' @param objective Optional objective used to determine required data
#'   components. One of `"poisson"`, `"weighted_poisson"`, `"logistic"`,
#'   `"weighted_logistic"`, `"iterative_poisson"`, or
#'   `"iterative_logistic"`.
#'
#' @details
#' Verifies structural invariants of `data`: `X` is a non-empty matrix or data frame,
#' `y` is binary (`0`/`1`), `locations` matches `nrow(X)`, and domain-specific
#' quantities required by `objective` (such as `quadrature_weight`, `dummy_intensity`,
#' `statistical_weight`, or `interaction_integral`) are present, finite, and
#' properly sized.
#'
#' @return Invisibly returns `TRUE` if the data object passes validation.
#'
#' @seealso [new_gbpp_data()], [prepare_gbpp()], [prepare_gbpp_network()],
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
#' validate_gbpp_data(dat, objective = "poisson")
#' @export
validate_gbpp_data <- function(data, objective = NULL) {
  if (!inherits(data, "gbpp_data")) {
    stop("`data` must be a `gbpp_data` object.", call. = FALSE)
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
