#' Calculate the planar spatial interaction integral
#'
#' Calculates the planar approximation \eqn{A = K(r_{\max}) - \pi r_{\max}^2}
#' to the spatial interaction integral used by the iteratively weighted GBPP
#' objectives.
#'
#' @param K_value Numeric scalar containing the estimated K-function value at `rmax`.
#' @param rmax Numeric scalar giving the maximum distance used for the K-function approximation.
#'
#' @details
#' Evaluates
#'
#' \deqn{
#' A = K(r_{\max}) - \pi r_{\max}^2,
#' }
#'
#' where \eqn{K(r_{\max})} is the estimated Ripley's K-function at distance
#' \eqn{r_{\max}}. For a second-order intensity-reweighted stationary planar
#' point process, the spatial interaction integral over a disk of radius
#' \eqn{r_{\max}} satisfies
#' \eqn{\int_{b(0, r_{\max})} (g(\|s\|) - 1)\,ds = K(r_{\max}) - \pi r_{\max}^2},
#' where \eqn{\pi r_{\max}^2} is the theoretical Poisson baseline value under
#' complete spatial randomness. Positive values indicate spatial clustering,
#' negative values indicate inhibition, and values near zero indicate weak or
#' no second-order interaction.
#'
#' @return A finite numeric scalar containing the estimated interaction integral.
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
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' @seealso [est_interaction()], [attach_interaction()], [calc_net_interaction()]
#'
#' @examples
#' calc_interaction(K_value = 0.05, rmax = 0.1)
#' @export
calc_interaction <- function(
  K_value,
  rmax
) {
  if (length(K_value) != 1L ||
    !is.numeric(K_value) ||
    !is.finite(K_value)) {
    stop(
      "`K_value` must be a single finite numeric value.",
      call. = FALSE
    )
  }

  if (length(rmax) != 1L ||
    !is.numeric(rmax) ||
    !is.finite(rmax) ||
    rmax <= 0) {
    stop(
      "`rmax` must be a single positive finite numeric value.",
      call. = FALSE
    )
  }

  interaction_integral <- K_value - pi * rmax^2

  if (!is.finite(interaction_integral)) {
    stop(
      "The interaction integral is not finite.",
      call. = FALSE
    )
  }

  interaction_integral
}

#' Estimate the planar interaction integral from an inhomogeneous K-function
#'
#' Estimates the spatial interaction integral used by the iteratively weighted
#' GBPP objectives by evaluating the inhomogeneous K-function at `rmax` and
#' subtracting the Poisson baseline \eqn{\pi r_{\max}^2}.
#'
#' @param point_pattern A planar `spatstat` point pattern (`ppp`).
#' @param rmax Maximum distance at which the inhomogeneous K-function is evaluated.
#' @param lambda Optional numeric vector of estimated intensity values
#'   at the observed points. If omitted, `Kinhom()` estimates the
#'   intensity using its default leave-one-out kernel procedure.
#' @param correction Edge correction used by `Kinhom()`. One of
#'   `"translate"` (default), `"isotropic"` (falls back to `"translate"` for
#'   pixel-mask windows), or `"border"`.
#'
#' @details
#' Calls [spatstat.explore::Kinhom()] at distance `rmax` using the requested edge
#' correction (`"translate"`, `"isotropic"`, or `"border"`) and transforms the
#' resulting estimate via [calc_interaction()],
#'
#' \deqn{
#' A = K(r_{\max}) - \pi r_{\max}^2.
#' }
#'
#' @return A finite numeric scalar containing the estimated interaction integral.
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
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' @seealso [calc_interaction()], [attach_interaction()], [est_net_interaction()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE)) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(30, win = win)
#'   est_interaction(pts, rmax = 0.15)
#' }
#' @export
est_interaction <- function(
  point_pattern,
  rmax,
  lambda = NULL,
  correction = c(
    "translate",
    "isotropic",
    "border"
  )
) {
  correction <- match.arg(correction)

  if (!inherits(point_pattern, "ppp")) {
    stop(
      "`point_pattern` must be a planar `spatstat` point pattern (`ppp`).",
      call. = FALSE
    )
  }

  if (correction == "isotropic" && spatstat.geom::is.mask(spatstat.geom::Window(point_pattern))) {
    correction <- "translate"
  }

  if (length(rmax) != 1L ||
    !is.numeric(rmax) ||
    !is.finite(rmax) ||
    rmax <= 0) {
    stop(
      "`rmax` must be a single positive finite numeric value.",
      call. = FALSE
    )
  }

  if (!is.null(lambda)) {
    if (!is.numeric(lambda)) {
      stop(
        "`lambda` must be numeric.",
        call. = FALSE
      )
    }

    if (length(lambda) != spatstat.geom::npoints(point_pattern)) {
      stop(
        "`lambda` must contain one value for each point in `point_pattern`.",
        call. = FALSE
      )
    }

    if (any(!is.finite(lambda)) ||
      any(lambda <= 0)) {
      stop(
        "`lambda` must contain positive finite values.",
        call. = FALSE
      )
    }
  }

  # Kinhom() requires at least two r values.
  r_values <- seq(
    from = 0,
    to = as.numeric(rmax),
    length.out = 2L
  )

  stopifnot(
    length(r_values) == 2L,
    is.numeric(r_values),
    identical(unname(r_values[1L]), 0),
    r_values[2L] > 0
  )

  K <- if (is.null(lambda)) {
    spatstat.explore::Kinhom(
      point_pattern,
      r = r_values,
      correction = correction
    )
  } else {
    spatstat.explore::Kinhom(
      point_pattern,
      lambda = lambda,
      r = r_values,
      correction = correction
    )
  }

  K_data <- as.data.frame(K)

  correction_column <- switch(correction,
    isotropic = "iso",
    translate = "trans",
    border = "border"
  )

  if (!correction_column %in% names(K_data)) {
    stop(
      paste0(
        "The requested K-function correction `",
        correction,
        "` was not returned by `Kinhom()`."
      ),
      call. = FALSE
    )
  }

  r_index <- which.min(
    abs(
      K_data$r - rmax
    )
  )

  K_value <- K_data[[correction_column]][r_index]

  if (length(K_value) != 1L ||
    !is.finite(K_value)) {
    stop(
      "The K-function estimate at `rmax` is not finite.",
      call. = FALSE
    )
  }

  calc_interaction(
    K_value = K_value,
    rmax = rmax
  )
}

#' Attach an estimated interaction integral to gbpp data
#'
#' Estimates the planar interaction integral from an observed point pattern
#' and attaches the resulting value to every row of a [`gbpp_data`] object.
#'
#' @param data A [`gbpp_data`] object.
#' @param point_pattern A `ppp` object containing the observed event locations.
#' @param rmax Maximum distance used in the K-function estimate.
#' @param lambda Optional intensity values supplied to `Kinhom()`.
#' @param correction Edge correction used by [spatstat.explore::Kinhom()]. One of
#'   `"translate"` (default), `"isotropic"`, or `"border"`.
#'
#' @details
#' Verifies that the event count `sum(data$y == 1)` matches `npoints(point_pattern)`,
#' computes the planar interaction integral via [est_interaction()], and stores
#' the recycled vector in `data$interaction_integral` for `"iterative_poisson"`
#' and `"iterative_logistic"` objectives.
#'
#' @return The modified [`gbpp_data`] object with `interaction_integral` populated.
#'
#' @references
#' Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for
#' inhomogeneous spatial point processes. *Biometrika*, 97(4), 867--880.
#' \doi{10.1093/biomet/asq047}
#'
#' @seealso [est_interaction()], [prepare_gbpp()], [attach_net_interaction()],
#'   [fit_gbpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE)) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(30, win = win)
#'   dat <- prepare_gbpp(pts, nd = 16L)
#'   dat <- attach_interaction(dat, pts, rmax = 0.15)
#'   head(dat$interaction_integral)
#' }
#' @export
attach_interaction <- function(
  data,
  point_pattern,
  rmax,
  lambda = NULL,
  correction = c("translate", "isotropic", "border")
) {
  correction <- match.arg(correction)

  if (!inherits(data, "gbpp_data")) {
    stop("`data` must be a `gbpp_data` object.", call. = FALSE)
  }

  if (!inherits(point_pattern, "ppp")) {
    stop("`point_pattern` must be a `ppp` object.", call. = FALSE)
  }

  n_events <- sum(data$y == 1)
  n_points <- spatstat.geom::npoints(point_pattern)

  if (n_events != n_points) {
    stop(
      paste0(
        "`data` contains ", n_events,
        " event(s), but `point_pattern` contains ",
        n_points, " point(s)."
      ),
      call. = FALSE
    )
  }

  interaction_integral <- est_interaction(
    point_pattern = point_pattern,
    rmax = rmax,
    lambda = lambda,
    correction = correction
  )

  data$interaction_integral <- rep(
    interaction_integral,
    length.out = length(data$y)
  )

  data
}
