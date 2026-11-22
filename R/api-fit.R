#' Fit a Gradient Boosted Point Process Model
#'
#' Fits a gradient boosting model to spatial point process data on planar
#' or linear network domains using point-process specific objectives and
#' modern gradient boosting backends (XGBoost, LightGBM, CatBoost).
#'
#' @param data An object of class `gbpp_data` constructed via [prepare_gbpp()]
#'   or [prepare_gbpp_network()]. Supports continuous numeric features as well as
#'   factor or character categorical covariates across all three engines.
#' @param engine Boosting backend engine: `"xgboost"`, `"lightgbm"`, or `"catboost"`.
#' @param objective Point process objective function: `"poisson"`,
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
#'   supplement defaults.
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
#' When `engine = "lightgbm"` and `nrow(data$X) < 20L`, `min_data_in_bin`,
#' `min_data_in_leaf`, and `min_sum_hessian_in_leaf` are automatically lowered to
#' `1L`, `1L`, and `1e-3` respectively (unless already smaller) so small spatial
#' fixtures can split leaves cleanly.
#'
#' @return An S3 object of class `c("gbpp_<engine>", "gbpp_model")` containing the
#'   fitted backend model object together with metadata attributes including
#'   `engine`, `objective`, `n_trees`, `baseline`, `feature_names`, and
#'   `factor_levels`. Supports [predict.gbpp_model()], [print.gbpp_model()],
#'   [plot.gbpp_model()], [explain_gbpp()], and [save_gbpp_model()].
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
#' @seealso [prepare_gbpp()], [prepare_gbpp_network()], [predict.gbpp_model()],
#'   [cv_gbpp()], [explain_gbpp()], [save_gbpp_model()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE) &&
#'     reticulate::py_available() &&
#'     reticulate::py_module_available("xgboost")) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(20, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_gbpp(pts, covariates = list(x = cov_x), nd = 16L)
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

  if (!inherits(data, "gbpp_data")) {
    stop("`data` must be a `gbpp_data` object.", call. = FALSE)
  }

  norm <- .gbpp_normalize_params(
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
    xgboost = .gbpp_fit_xgboost(
      data = data,
      objective = objective,
      params = norm$params,
      n_trees = norm$n_trees,
      baseline = baseline,
      ...
    ),
    lightgbm = .gbpp_fit_lightgbm(
      data = data,
      objective = objective,
      params = norm$params,
      n_trees = norm$n_trees,
      baseline = baseline,
      ...
    ),
    catboost = .gbpp_fit_catboost(
      data = data,
      objective = objective,
      params = norm$params,
      n_trees = norm$n_trees,
      baseline = baseline,
      ...
    )
  )
}
