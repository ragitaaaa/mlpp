#' mlpp: Machine Learning for Point Processes
#'
#' Provides spatial point process modeling via machine learning methods
#' (currently supporting gradient boosted decision trees) on planar windows
#' (`ppp`) and linear networks (`lpp`), integrating `spatstat` quadrature schemes
#' and interaction diagnostics with Python boosting backends (`xgboost`, `lightgbm`,
#' `catboost`) through `reticulate`.
#'
#' @details
#' Main entry points in the package workflow:
#' - **Data preparation**: [prepare_gbpp()], [prepare_gbpp_network()], [prepare_quad_data()],
#'   [new_gbpp_data()], [validate_gbpp_data()].
#' - **Interaction & iterative weights**: [est_interaction()], [est_net_interaction()],
#'   [attach_interaction()], [attach_net_interaction()], [attach_stat_weight()],
#'   [calc_iter_poisson_weight()], [calc_iter_logistic_weight()].
#' - **Model fitting & prediction**: [fit_gbpp()], [predict.gbpp_model()], [plot.gbpp_model()],
#'   [save_gbpp_model()], [load_gbpp_model()].
#' - **Diagnostics & cross-validation**: [cv_gbpp()], [explain_gbpp()],
#'   [calc_poisson_loglik()], [calc_poisson_deviance()].
#' - **Python backend setup**: [configure_gbpp_python()], [install_gbpp_python()],
#'   [check_gbpp_python()], [get_gbpp_python()].
#'
#' @references
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
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
#' @seealso [prepare_gbpp()], [prepare_gbpp_network()], [fit_gbpp()],
#'   [cv_gbpp()], [explain_gbpp()]
#'
#' @importFrom stats complete.cases predict sd
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end

## mockable bindings: start
## mockable bindings: end
NULL
