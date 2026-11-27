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
#' - **Data preparation**: [prepare_mlpp()], [mlpp_data()].
#' - **Weights & interaction**: [attach_weights()], [attach_interaction()].
#' - **Model fitting & prediction**: [fit_gbpp()], [predict.mlpp_model()],
#'   [plot.mlpp_model()], [save_mlpp()], [load_mlpp()].
#' - **Evaluation, cross-validation & explainability**: [eval_mlpp()], [cv_mlpp()],
#'   [explain_mlpp()].
#' - **Python backend setup**: [install_mlpp_python()], [check_mlpp_python()].
#'
#' @references
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
#' *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
#' Discovery and Data Mining*, 785--794. \doi{10.1145/2939672.2939785}
#'
#' Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu,
#' T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree.
#' *Advances in Neural Information Processing Systems*, 30, 3146--3154.
#'
#' Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A.
#' (2018). CatBoost: Unbiased boosting with categorical features. *Advances in
#' Neural Information Processing Systems*, 31, 6638--6648.
#' \doi{10.48550/arXiv.1706.09516}
#'
#' Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B.,
#' Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020).
#' From local explanations to global understanding with explainable AI for trees.
#' *Nature Machine Intelligence*, 2(1), 56--67.
#' \doi{10.1038/s42256-019-0138-9}
#'
#' Cronie, O., Moradi, M., & Biscio, C. A. N. (2024). A cross-validation-based
#' statistical theory for point processes. *Biometrika*, 111(2), 625--641.
#' \doi{10.1093/biomet/asad041}
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
#' @seealso [prepare_mlpp()], [mlpp_data()], [fit_gbpp()], [cv_mlpp()],
#'   [explain_mlpp()]
#'
#' @importFrom stats complete.cases predict sd
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end

## mockable bindings: start
## mockable bindings: end
NULL
