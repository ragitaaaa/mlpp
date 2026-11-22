#' Calculate gradient and Hessian for a GBPP objective (Reference Implementation)
#'
#' Pure R reference implementations of the first and second derivatives
#' of the point process likelihoods. Used for verification and unit tests.
#'
#' @param data A `gbpp_data` object.
#' @param prediction Numeric vector of current predicted log-intensities
#'   (\eqn{\eta_i}).
#' @param objective Objective to use. One of `"poisson"`, `"weighted_poisson"`,
#'   `"logistic"`, `"weighted_logistic"`, `"iterative_poisson"`, or
#'   `"iterative_logistic"`.
#'
#' @details
#' Evaluates the exact first derivative (`gradient`) and second derivative /
#' Fisher scoring curvature (`hessian`) of the chosen point process loss with
#' respect to the log-intensity predictor \eqn{\eta_i = \log \lambda_i} (denoted
#' \eqn{\phi(u_i; \mathbf{f})} in Lu et al., 2025 and Choiruddin et al., 2026),
#' clamped to `[-50, 50]` for numerical stability:
#' - `"poisson"`: \eqn{g_i = v_i \lambda_i - y_i}, \eqn{h_i = v_i \lambda_i},
#'   where \eqn{v_i} is `quadrature_weight`.
#' - `"weighted_poisson"`: \eqn{g_i = w_i (v_i \lambda_i - y_i)},
#'   \eqn{h_i = w_i v_i \lambda_i}, where \eqn{w_i} is `statistical_weight`.
#' - `"logistic"`: uses `dummy_intensity` \eqn{\delta_i} via
#'   \eqn{\pi_i = \lambda_i / (\lambda_i + \delta_i)} and yields
#'   \eqn{g_i = \pi_i - y_i}, \eqn{h_i = \pi_i (1 - \pi_i)}.
#' - `"weighted_logistic"`: \eqn{g_i = w_i (\pi_i - y_i)},
#'   \eqn{h_i = w_i \pi_i (1 - \pi_i)} with `statistical_weight` \eqn{w_i}.
#' - `"iterative_poisson"` / `"iterative_logistic"`: incorporates unit-mean
#'   normalized dependence weights \eqn{w_i} from [calc_iter_poisson_weight()]
#'   or [calc_iter_logistic_weight()] using `interaction_integral` \eqn{A_i}.
#'
#' @return A list containing equal-length numeric vectors `gradient` and `hessian`
#'   (and, for `"iterative_poisson"` and `"iterative_logistic"`, the unit-mean
#'   normalized `weight` vector).
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
#' Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for
#' inhomogeneous spatial point processes. *Biometrika*, 97(4), 867--880.
#' \doi{10.1093/biomet/asq047}
#'
#' Lu, C., Guan, Y., van Lieshout, M.-C., & Xu, G. (2025). XGBoostPP: Tree-based
#' estimation of point process intensity functions. *Journal of Computational and
#' Graphical Statistics*, 35(1), 229--240. \doi{10.1080/10618600.2025.2520582}
#'
#' Choiruddin, A., Tasbihi, N. L., Abbad, M. A., Syukron, M., Widhianingsih,
#' T. D. A., & Mateu, J. (2026). XGBoost algorithm for a spatial point process
#' intensity estimation based on logistic loss. *Computational Journal of
#' Mathematical and Statistical Sciences*, 5(2), 472--496.
#' \doi{10.21608/cjmss.2026.452469.1353}
#'
#' @seealso [fit_gbpp()], [calc_iter_poisson_weight()], [calc_iter_logistic_weight()]
#'
#' @examples
#' dat <- new_gbpp_data(
#'   X = data.frame(x = c(0.2, 0.8, 0.5)),
#'   y = c(1, 0, 0),
#'   locations = data.frame(x = c(0.2, 0.8, 0.5), y = c(0.3, 0.7, 0.5)),
#'   quadrature_weight = c(0.2, 0.4, 0.4),
#'   type = "planar"
#' )
#' derivs <- calc_objective(dat, prediction = c(0, 0, 0), objective = "poisson")
#' derivs$gradient
#' @export
calc_objective <- function(
  data,
  prediction,
  objective = c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  )
) {
  objective <- match.arg(objective)

  validate_gbpp_data(
    data,
    objective = objective
  )

  if (!is.numeric(prediction)) {
    stop("`prediction` must be a numeric vector.", call. = FALSE)
  }

  if (length(prediction) != length(data$y)) {
    stop("`prediction` must have one value for each observation.", call. = FALSE)
  }

  if (any(!is.finite(prediction))) {
    stop("`prediction` must contain only finite values.", call. = FALSE)
  }

  switch(objective,
    poisson = .gbpp_deriv_poisson(data = data, prediction = prediction),
    weighted_poisson = .gbpp_deriv_weighted_poisson(data = data, prediction = prediction),
    logistic = .gbpp_deriv_logistic(data = data, prediction = prediction),
    weighted_logistic = .gbpp_deriv_weighted_logistic(data = data, prediction = prediction),
    iterative_poisson = .gbpp_deriv_iter_poisson(data = data, prediction = prediction),
    iterative_logistic = .gbpp_deriv_iter_logistic(data = data, prediction = prediction)
  )
}

#' @keywords internal
#' @noRd
.gbpp_deriv_poisson <- function(data, prediction) {
  phi <- pmin(pmax(prediction, -50), 50)
  lambda <- exp(phi)
  gradient <- data$quadrature_weight * lambda - data$y
  hessian <- data$quadrature_weight * lambda

  list(
    gradient = gradient,
    hessian = hessian
  )
}

#' @keywords internal
#' @noRd
.gbpp_deriv_weighted_poisson <- function(data, prediction) {
  phi <- pmin(pmax(prediction, -50), 50)
  lambda <- exp(phi)

  gradient <- (data$quadrature_weight * data$statistical_weight * lambda) -
    (data$y * data$statistical_weight)
  hessian <- data$quadrature_weight * data$statistical_weight * lambda

  list(
    gradient = gradient,
    hessian = hessian
  )
}

#' @keywords internal
#' @noRd
.gbpp_deriv_logistic <- function(data, prediction) {
  phi <- pmin(pmax(prediction, -50), 50)
  delta <- data$dummy_intensity
  pi <- stats::plogis(phi - log(delta))
  gradient <- pi - data$y
  hessian <- pi * (1 - pi)

  list(
    gradient = gradient,
    hessian = hessian
  )
}

#' @keywords internal
#' @noRd
.gbpp_deriv_weighted_logistic <- function(data, prediction) {
  phi <- pmin(pmax(prediction, -50), 50)
  delta <- data$dummy_intensity
  pi <- stats::plogis(phi - log(delta))

  gradient <- data$statistical_weight * (pi - data$y)
  hessian <- data$statistical_weight * pi * (1 - pi)

  list(
    gradient = gradient,
    hessian = hessian
  )
}

#' @keywords internal
#' @noRd
.gbpp_deriv_iter_poisson <- function(data, prediction, clamp = TRUE) {
  phi <- pmin(pmax(prediction, -50), 50)
  lambda <- exp(phi)

  weight <- calc_iter_poisson_weight(
    prediction = phi,
    interaction_integral = data$interaction_integral,
    clamp = clamp
  )

  gradient <- weight * (data$quadrature_weight * lambda - data$y)
  hessian <- weight * data$quadrature_weight * lambda

  list(
    gradient = gradient,
    hessian = hessian,
    weight = weight
  )
}

#' @keywords internal
#' @noRd
.gbpp_deriv_iter_logistic <- function(data, prediction, clamp = TRUE) {
  phi <- pmin(pmax(prediction, -50), 50)
  lambda <- exp(phi)
  delta <- data$dummy_intensity

  weight <- calc_iter_logistic_weight(
    prediction = phi,
    dummy_intensity = delta,
    interaction_integral = data$interaction_integral,
    clamp = clamp
  )

  pi <- stats::plogis(phi - log(delta))

  gradient <- weight * (pi - data$y)
  hessian <- weight * pi * (1 - pi)

  list(
    gradient = gradient,
    hessian = hessian,
    weight = weight
  )
}
