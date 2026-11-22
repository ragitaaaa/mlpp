#' Extract dummy-process intensity from a logistic quadrature scheme
#'
#' Extracts the dummy point intensity vector stored in a `spatstat` logistic
#' quadrature scheme (`quadscheme.logi`).
#'
#' @param Q A logistic quadrature scheme from `spatstat` (`inherits(Q, "quad")`).
#'
#' @details
#' Reads `Q$param$rho`, validates that all entries are positive finite numbers,
#' and recycles scalar `rho` across all quadrature points in `Q`. Throws an error
#' when `Q$param$rho` is absent (e.g., standard Berman-Turner `quadscheme()`).
#'
#' @return A numeric vector with one positive finite dummy-process intensity value
#'   for each quadrature point in `Q`.
#'
#' @references
#' Baddeley, A., Coeurjolly, J.-F., Rubak, E., & Waagepetersen, R. (2014).
#' Logistic regression for spatial Gibbs point processes. *Biometrika*,
#' 101(2), 377--392. \doi{10.1093/biomet/ast056}
#'
#' @seealso [calc_dummy_intensity()], [calc_net_dummy_intensity()], [prepare_gbpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE)) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(15, win = win)
#'   Q <- spatstat.geom::quadscheme.logi(pts, dummytype = "grid", nd = 8)
#'   rho <- extract_dummy_intensity(Q)
#'   head(rho)
#' }
#' @export
extract_dummy_intensity <- function(Q) {
  if (!inherits(Q, "quad")) {
    stop(
      "`Q` must be a spatstat quadrature scheme.",
      call. = FALSE
    )
  }

  n <- length(spatstat.geom::is.data(Q))
  rho <- Q$param$rho

  if (is.null(rho)) {
    stop(
      paste0(
        "Dummy-process intensity could not be extracted from `Q`. ",
        "Provide it explicitly for quadrature schemes that do not ",
        "store `rho` in `Q$param$rho`."
      ),
      call. = FALSE
    )
  }

  if (!is.numeric(rho)) {
    stop(
      "`Q$param$rho` must be numeric.",
      call. = FALSE
    )
  }

  if (any(!is.finite(rho)) || any(rho <= 0)) {
    stop(
      "Dummy-process intensity must contain positive finite values.",
      call. = FALSE
    )
  }

  if (length(rho) == 1L) {
    rho <- rep(rho, n)
  }

  if (length(rho) != n) {
    stop(
      paste0(
        "Dummy-process intensity must either be a single value ",
        "or have one value for each quadrature point."
      ),
      call. = FALSE
    )
  }

  rho
}

#' Calculate the effective dummy intensity for a planar quadrature scheme
#'
#' Calculates the homogeneous dummy intensity \eqn{\delta = m / |W|} implied by
#' a planar Berman-Turner quadrature scheme, where \eqn{m} is the number of
#' dummy points and \eqn{|W|} is the total window area.
#'
#' @param y Binary response vector with `1` for data points and `0` for
#'   dummy points.
#' @param quadrature_weight Positive Berman-Turner quadrature weights, or
#'   an observation window (`owin`).
#'
#' @details
#' Evaluates
#'
#' \deqn{
#' \delta = \frac{m}{|W|},
#' }
#'
#' counting the number of dummy locations \eqn{m = \sum_i (1 - y_i)} and
#' computing total area \eqn{|W|} either via
#' `spatstat.geom::area(quadrature_weight)` when an `owin` is passed, or via
#' `sum(quadrature_weight)` when quadrature weights are supplied.
#'
#' @return A positive numeric scalar giving the effective homogeneous dummy
#'   intensity \eqn{\delta = m / |W|} per unit area.
#'
#' @references
#' Baddeley, A., Coeurjolly, J.-F., Rubak, E., & Waagepetersen, R. (2014).
#' Logistic regression for spatial Gibbs point processes. *Biometrika*,
#' 101(2), 377--392. \doi{10.1093/biomet/ast056}
#'
#' @seealso [extract_dummy_intensity()], [calc_net_dummy_intensity()], [prepare_gbpp()]
#'
#' @examples
#' y <- c(1, 0, 0, 0, 0)
#' w <- c(0.1, 0.225, 0.225, 0.225, 0.225)
#' calc_dummy_intensity(y, w)
#' @export
calc_dummy_intensity <- function(
  y,
  quadrature_weight
) {
  if (!is.numeric(y) ||
    length(y) == 0L ||
    anyNA(y) ||
    any(!is.finite(y)) ||
    any(!y %in% c(0, 1))) {
    stop(
      "`y` must be a non-empty finite binary numeric vector.",
      call. = FALSE
    )
  }

  if (inherits(quadrature_weight, "owin")) {
    window_area <- spatstat.geom::area(quadrature_weight)
  } else {
    if (!is.numeric(quadrature_weight) ||
      length(quadrature_weight) != length(y) ||
      anyNA(quadrature_weight) ||
      any(!is.finite(quadrature_weight)) ||
      any(quadrature_weight <= 0)) {
      stop(
        "`quadrature_weight` must be a positive finite numeric vector with ",
        "the same length as `y`, or an `owin` window.",
        call. = FALSE
      )
    }
    window_area <- sum(quadrature_weight)
  }

  n_dummy <- sum(y == 0)

  if (n_dummy <= 0) {
    stop(
      "At least one dummy point is required.",
      call. = FALSE
    )
  }

  if (!is.finite(window_area) || window_area <= 0) {
    stop(
      "The total window area must be positive and finite.",
      call. = FALSE
    )
  }

  dummy_intensity <- n_dummy / window_area

  if (!is.finite(dummy_intensity) || dummy_intensity <= 0) {
    stop(
      "Failed to calculate a positive finite dummy intensity.",
      call. = FALSE
    )
  }

  dummy_intensity
}

#' Calculate the effective dummy intensity for a network quadrature scheme
#'
#' Calculates the homogeneous dummy intensity \eqn{\delta = m / |\mathcal{L}|}
#' implied by a linear-network quadrature scheme, where \eqn{m} is the number
#' of dummy points and \eqn{|\mathcal{L}|} is the total network length.
#'
#' @param y Binary response vector with `1` for data points and `0` for
#'   dummy points.
#' @param quadrature_weight Positive Berman-Turner quadrature weights.
#'
#' @details
#' Evaluates
#'
#' \deqn{
#' \delta = \frac{m}{|\mathcal{L}|},
#' }
#'
#' computing total network length \eqn{|\mathcal{L}| = \sum_i v_i} from the
#' supplied `quadrature_weight` vector and dividing the count of dummy quadrature
#' points \eqn{m = \sum_i (1 - y_i)} by \eqn{|\mathcal{L}|} to produce an
#' approximate homogeneous network dummy intensity for logistic objectives on
#' linear networks.
#'
#' @return A positive numeric scalar giving the effective homogeneous dummy
#'   intensity \eqn{\delta = m / |\mathcal{L}|} per unit network length.
#'
#' @references
#' Baddeley, A., Nair, G., Rakshit, S., McSwiggan, G., & Davies, T. M. (2021).
#' Analysing point patterns on networks --- A review. *Spatial Statistics*,
#' 42, 100435. \doi{10.1016/j.spasta.2020.100435}
#'
#' @seealso [calc_dummy_intensity()], [extract_dummy_intensity()],
#'   [prepare_gbpp_network()]
#'
#' @examples
#' y <- c(1, 0, 0, 0)
#' w <- c(0.5, 1.5, 1.5, 1.5)
#' calc_net_dummy_intensity(y, w)
#' @export
calc_net_dummy_intensity <- function(
  y,
  quadrature_weight
) {
  if (!is.numeric(y) ||
    length(y) == 0L ||
    anyNA(y) ||
    any(!is.finite(y)) ||
    any(!y %in% c(0, 1))) {
    stop(
      "`y` must be a non-empty finite binary numeric vector.",
      call. = FALSE
    )
  }

  if (!is.numeric(quadrature_weight) ||
    length(quadrature_weight) != length(y) ||
    anyNA(quadrature_weight) ||
    any(!is.finite(quadrature_weight)) ||
    any(quadrature_weight <= 0)) {
    stop(
      "`quadrature_weight` must be a positive finite numeric vector with the same length as `y`.",
      call. = FALSE
    )
  }

  n_dummy <- sum(y == 0)
  network_length <- sum(quadrature_weight)

  if (n_dummy <= 0) {
    stop(
      "At least one dummy point is required.",
      call. = FALSE
    )
  }

  if (!is.finite(network_length) || network_length <= 0) {
    stop(
      "The total quadrature length must be positive and finite.",
      call. = FALSE
    )
  }

  dummy_intensity <- n_dummy / network_length

  if (!is.finite(dummy_intensity) || dummy_intensity <= 0) {
    stop(
      "Failed to calculate a positive finite dummy intensity.",
      call. = FALSE
    )
  }

  dummy_intensity
}
