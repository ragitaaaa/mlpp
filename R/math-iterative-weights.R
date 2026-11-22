#' Calculate iteratively updated Poisson weights
#'
#' Calculates the spatial dependence-adjusted weights used by the
#' iteratively weighted Poisson formulation under an IRLS (frozen-weight)
#' gradient-boosting approximation, normalized to unit sample mean.
#'
#' @param prediction Numeric vector of current predicted log-intensities
#'   (\eqn{\eta_i}).
#' @param interaction_integral Numeric vector containing the spatial
#'   interaction integral \eqn{A_i} for each quadrature point.
#' @param clamp Logical; if `TRUE`, clamps `1 + lambda * interaction_integral`
#'   at `min_denom` (matching Python runtime safeguards) instead of erroring
#'   on non-positive denominators. Default is `FALSE`.
#' @param min_denom Positive scalar floor applied when `clamp = TRUE`.
#'
#' @details
#' For a current log-intensity prediction \eqn{\eta_i}, the corresponding
#' intensity is \eqn{\lambda_i = \exp(\eta_i)}, yielding the unnormalized and
#' unit-mean normalized Guan & Shen (2010) / Lu et al. (2025) optimal spatial
#' dependence weights:
#'
#' \deqn{
#' \tilde{w}_i = \frac{1}{1 + \lambda_i A_i},
#' \qquad
#' w_i = \frac{\tilde{w}_i}{\frac{1}{N} \sum_{r=1}^{N} \tilde{w}_r},
#' }
#'
#' where \eqn{A_i} denotes the spatial interaction integral and \eqn{N = n + m}
#' is the total number of quadrature points. When `clamp = TRUE`,
#' \eqn{1 + \lambda_i A_i} is replaced by
#' \eqn{\max(\mathrm{min\_denom}, 1 + \lambda_i A_i)}. Normalizing to unit mean
#' preserves effective quadrature scale across boosting iterations.
#'
#' @return A positive numeric vector of iteratively updated weights normalized to
#'   unit mean (`mean(weight) == 1`).
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
#' @seealso [calc_iter_logistic_weight()], [est_interaction()], [est_net_interaction()]
#'
#' @examples
#' pred <- c(-0.5, 0.0, 0.5)
#' A <- c(0.1, 0.1, 0.1)
#' calc_iter_poisson_weight(pred, A)
#' @export
calc_iter_poisson_weight <- function(
  prediction,
  interaction_integral,
  clamp = FALSE,
  min_denom = 1e-4
) {
  if (!is.numeric(prediction)) {
    stop("`prediction` must be numeric.", call. = FALSE)
  }

  if (!is.numeric(interaction_integral)) {
    stop("`interaction_integral` must be numeric.", call. = FALSE)
  }

  if (length(prediction) != length(interaction_integral)) {
    stop("`prediction` and `interaction_integral` must have the same length.", call. = FALSE)
  }

  if (any(!is.finite(prediction))) {
    stop("`prediction` must contain only finite values.", call. = FALSE)
  }

  if (any(!is.finite(interaction_integral))) {
    stop("`interaction_integral` must contain only finite values.", call. = FALSE)
  }

  phi <- pmin(pmax(prediction, -50), 50)
  lambda <- exp(phi)

  denominator <- 1 + lambda * interaction_integral

  if (isTRUE(clamp)) {
    denominator <- pmax(min_denom, denominator)
  } else if (any(denominator <= 0)) {
    stop(
      paste0(
        "The iterative Poisson weight is undefined for at least ",
        "one observation because `1 + lambda * interaction_integral` ",
        "is not positive."
      ),
      call. = FALSE
    )
  }

  weight <- 1 / denominator
  weight / mean(weight)
}

#' Calculate iteratively updated logistic weights
#'
#' Calculates the spatial dependence-adjusted weights used by the
#' iteratively weighted logistic formulation under an IRLS (frozen-weight)
#' gradient-boosting approximation, normalized to unit sample mean.
#'
#' @param prediction Numeric vector of current predicted log-intensities
#'   (\eqn{\eta_i}).
#' @param dummy_intensity Numeric vector of dummy-process intensities
#'   (\eqn{\delta_i}).
#' @param interaction_integral Numeric vector containing the spatial
#'   interaction integral \eqn{A_i} for each quadrature point.
#' @param clamp Logical; if `TRUE`, clamps `1 + lambda * interaction_integral`
#'   at `min_denom` (matching Python runtime safeguards) instead of erroring
#'   on non-positive denominators. Default is `FALSE`.
#' @param min_denom Positive scalar floor applied when `clamp = TRUE`.
#'
#' @details
#' For a current log-intensity prediction \eqn{\eta_i} with
#' \eqn{\lambda_i = \exp(\eta_i)}, dummy-process intensity \eqn{\delta_i}, and
#' interaction integral \eqn{A_i}, the unnormalized and unit-mean normalized
#' weights (Choiruddin et al., 2026) are
#'
#' \deqn{
#' \tilde{w}_i = \frac{\lambda_i + \delta_i}{\delta_i (1 + \lambda_i A_i)},
#' \qquad
#' w_i = \frac{\tilde{w}_i}{\frac{1}{N} \sum_{r=1}^{N} \tilde{w}_r},
#' }
#'
#' where \eqn{N = n + m} is the total number of quadrature points. When
#' `clamp = TRUE`, \eqn{1 + \lambda_i A_i} is replaced by
#' \eqn{\max(\mathrm{min\_denom}, 1 + \lambda_i A_i)}. Normalizing to unit mean
#' (`mean(weight) == 1`) preserves effective quadrature scale across iterations.
#'
#' @return A positive numeric vector of iteratively updated logistic weights
#'   normalized to unit mean (`mean(weight) == 1`).
#'
#' @references
#' Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for
#' inhomogeneous spatial point processes. *Biometrika*, 97(4), 867--880.
#' \doi{10.1093/biomet/asq047}
#'
#' Baddeley, A., Coeurjolly, J.-F., Rubak, E., & Waagepetersen, R. (2014).
#' Logistic regression for spatial Gibbs point processes. *Biometrika*,
#' 101(2), 377--392. \doi{10.1093/biomet/ast056}
#'
#' Choiruddin, A., Tasbihi, N. L., Abbad, M. A., Syukron, M., Widhianingsih,
#' T. D. A., & Mateu, J. (2026). XGBoost algorithm for a spatial point process
#' intensity estimation based on logistic loss. *Computational Journal of
#' Mathematical and Statistical Sciences*, 5(2), 472--496.
#' \doi{10.21608/cjmss.2026.452469.1353}
#'
#' @seealso [calc_iter_poisson_weight()], [est_interaction()], [est_net_interaction()]
#'
#' @examples
#' pred <- c(-0.5, 0.0, 0.5)
#' delta <- c(5.0, 5.0, 5.0)
#' A <- c(0.1, 0.1, 0.1)
#' calc_iter_logistic_weight(pred, delta, A)
#' @export
calc_iter_logistic_weight <- function(
  prediction,
  dummy_intensity,
  interaction_integral,
  clamp = FALSE,
  min_denom = 1e-4
) {
  if (!is.numeric(prediction)) {
    stop("`prediction` must be numeric.", call. = FALSE)
  }

  if (!is.numeric(dummy_intensity)) {
    stop("`dummy_intensity` must be numeric.", call. = FALSE)
  }

  if (!is.numeric(interaction_integral)) {
    stop("`interaction_integral` must be numeric.", call. = FALSE)
  }

  if (length(prediction) != length(dummy_intensity) ||
    length(prediction) != length(interaction_integral)) {
    stop(
      "`prediction`, `dummy_intensity`, and `interaction_integral` must ",
      "have the same length.",
      call. = FALSE
    )
  }

  if (any(!is.finite(prediction))) {
    stop("`prediction` must contain only finite values.", call. = FALSE)
  }

  if (any(!is.finite(dummy_intensity)) || any(dummy_intensity <= 0)) {
    stop("`dummy_intensity` must contain positive finite values.", call. = FALSE)
  }

  if (any(!is.finite(interaction_integral))) {
    stop("`interaction_integral` must contain only finite values.", call. = FALSE)
  }

  phi <- pmin(pmax(prediction, -50), 50)
  lambda <- exp(phi)

  denom_int <- 1 + lambda * interaction_integral
  if (isTRUE(clamp)) {
    denom_int <- pmax(min_denom, denom_int)
  }
  denominator <- dummy_intensity * denom_int

  if (!isTRUE(clamp) && any(denominator <= 0)) {
    stop(
      paste0(
        "The iterative Logistic weight is undefined for at least ",
        "one observation because the denominator is not positive."
      ),
      call. = FALSE
    )
  }

  weight <- (lambda + dummy_intensity) / denominator
  weight / mean(weight)
}
