#' Fit a Gradient Boosted Point Process Model
#'
#' Fits a gradient boosting model to spatial point process data on planar
#' or linear network domains using point-process specific objectives and
#' modern gradient boosting backends (XGBoost, LightGBM, CatBoost).
#'
#' @param data An object of class `mlpp_data` constructed via [prepare_mlpp()]
#'   or [mlpp_data()]. Supports continuous numeric features as well as
#'   factor or character categorical covariates across all three engines.
#' @param engine Boosting backend engine: `"xgboost"` (default), `"lightgbm"`,
#'   or `"catboost"`.
#' @param objective Point process objective function: `"poisson"` (default),
#'   `"weighted_poisson"`, `"logistic"`, `"weighted_logistic"`,
#'   `"iterative_poisson"`, or `"iterative_logistic"`.
#' @param n_trees Number of boosting rounds (trees). Default is `100L`.
#' @param learning_rate Learning rate (shrinkage factor). Default is `0.05`.
#' @param max_depth Maximum tree depth. Default is `6L`.
#' @param subsample Row subsampling ratio in `(0, 1]`. Default is `1.0`.
#' @param colsample_bytree Column (feature) subsampling ratio in `(0, 1]`. Default is `1.0`.
#' @param random_seed Random seed for reproducibility. Default is `123L`.
#' @param verbose Logical flag indicating whether training progress is printed. Default is `FALSE`.
#' @param baseline Optional numeric scalar or observation-length vector for the
#'   initial log-intensity intercept \eqn{\eta_0}. If `NULL` (default), it is
#'   initialized to \eqn{\log(\sum_i y_i / \sum_i v_i)} (where \eqn{v_i}
#'   denotes the quadrature weights) for `"poisson"`, `"weighted_poisson"`, and
#'   `"iterative_poisson"` objectives, and to `0` for `"logistic"`,
#'   `"weighted_logistic"`, and `"iterative_logistic"`.
#' @param params Named list of backend-specific parameters to override or
#'   supplement defaults. Common hyperparameter aliases (e.g., `learning_rate`,
#'   `max_depth`, `subsample`, `colsample_bytree`, `random_seed`) are mapped
#'   to each backend automatically.
#' @param ... Additional arguments passed to the backend fit call.
#'
#' @details
#' `fit_gbpp()` models the inhomogeneous point process intensity \eqn{\lambda(u)}
#' on a planar window \eqn{W} or linear network \eqn{\mathcal{L}} on the log
#' scale,
#' \deqn{\log \lambda(u) = \eta_0 + F(z(u)), \qquad F(z(u)) = \sum_{k=1}^{K} f_k(z(u)),}
#' where \eqn{F(z(u))} is an ensemble of \eqn{K} (`n_trees`) regression trees
#' evaluated on covariate vector \eqn{z(u) \in \mathbb{R}^p} and fitted via
#' Python backends (`xgboost`, `lightgbm`, or `catboost`) using custom first- and
#' second-order derivatives (`grad`, `hess`) derived from quadrature-weighted
#' point process estimating equations.
#'
#' Available objectives correspond to Berman-Turner Poisson likelihood approximations
#' (`"poisson"`), spatially weighted Poisson composite likelihood (`"weighted_poisson"`),
#' Baddeley et al. (2014) logistic regression (`"logistic"`, `"weighted_logistic"`),
#' and iteratively reweighted schemes (`"iterative_poisson"`, `"iterative_logistic"`).
#'
#' Categorical covariates are encoded natively via Ordered Target Statistics
#' (`cat_features`) in CatBoost and as consistent training-level integer codes in
#' XGBoost and LightGBM. When `engine = "lightgbm"` and `nrow(data$X) < 20L`,
#' `min_data_in_bin`, `min_data_in_leaf`, and `min_sum_hessian_in_leaf` are
#' automatically lowered to `1L`, `1L`, and `1e-3` respectively (unless already
#' smaller) so small spatial fixtures can split leaves cleanly.
#'
#' In CatBoost (`engine = "catboost"`), stateless objectives (`"poisson"`,
#' `"weighted_poisson"`, `"logistic"`, `"weighted_logistic"`) support `Ordered`
#' boosting, `Bernoulli` subsampling, and multi-threading, whereas stateful
#' iterative objectives (`"iterative_poisson"`, `"iterative_logistic"`) require
#' full-batch `boosting_type = "Plain"` (`has_time = TRUE`, `thread_count = 1L`,
#' `subsample = 1.0`).
#'
#' @return An S3 object inheriting from `c("gbpp_<engine>", "gbpp_model", "mlpp_model")`
#'   wrapping the fitted Python booster together with metadata attributes
#'   `engine`, `objective`, `baseline`, `feature_names`, `factor_levels`, `params`,
#'   `n_trees`, `serialized_model`, `serialized_format`, `data_info` (and
#'   `cat_features` for CatBoost). Supports [predict.mlpp_model()],
#'   [print.mlpp_model()], [plot.mlpp_model()], [eval_mlpp()], [explain_mlpp()],
#'   and [save_mlpp()].
#'
#' @references
#' Berman, M., & Turner, T. R. (1992). Approximating point process likelihoods
#' with GLIM. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*,
#' 41(1), 31--38. \doi{10.2307/2347614}
#'
#' Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for
#' inhomogeneous spatial point processes. *Biometrika*, 97(4), 867--880.
#' \doi{10.1093/biomet/asq047}
#'
#' Baddeley, A., Coeurjolly, J.-F., Rubak, E., & Waagepetersen, R. (2014).
#' Logistic regression for spatial Gibbs point processes. *Biometrika*,
#' 101(2), 377--392. \doi{10.1093/biomet/ast056}
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
#' @seealso [prepare_mlpp()], [mlpp_data()], [predict.mlpp_model()],
#'   [eval_mlpp()], [cv_mlpp()], [explain_mlpp()], [save_mlpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'     reticulate::py_available() &&
#'     reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   fit <- fit_gbpp(dat, engine = "xgboost", objective = "poisson", n_trees = 10L)
#'   print(fit)
#' }
#' @export
fit_gbpp <- function(
    data,
    engine = c("xgboost", "lightgbm", "catboost"),
    objective = c(
      "poisson",
      "weighted_poisson",
      "logistic",
      "weighted_logistic",
      "iterative_poisson",
      "iterative_logistic"
    ),
    n_trees = 100L,
    learning_rate = 0.05,
    max_depth = 6L,
    subsample = 1.0,
    colsample_bytree = 1.0,
    random_seed = 123L,
    verbose = FALSE,
    baseline = NULL,
    params = list(),
    ...
) {
  engine <- match.arg(engine)
  objective <- match.arg(objective)

  if (!inherits(data, "mlpp_data")) {
    stop("`data` must be an `mlpp_data` object.", call. = FALSE)
  }

  norm <- .norm_params(
    engine = engine,
    n_trees = n_trees,
    learning_rate = learning_rate,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    random_seed = random_seed,
    verbose = verbose,
    user_params = params
  )

  switch(
    engine,
    xgboost = .fit_xgb(
      data = data,
      objective = objective,
      params = norm$params,
      n_trees = norm$n_trees,
      baseline = baseline,
      ...
    ),
    lightgbm = .fit_lgb(
      data = data,
      objective = objective,
      params = norm$params,
      n_trees = norm$n_trees,
      baseline = baseline,
      ...
    ),
    catboost = .fit_cb(
      data = data,
      objective = objective,
      params = norm$params,
      n_trees = norm$n_trees,
      baseline = baseline,
      ...
    )
  )
}
