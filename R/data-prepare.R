#' Prepare an mlpp spatial point-process data object
#'
#' Unified data preparation entry point supporting 2D planar point patterns
#' (`ppp`), 1D linear network point patterns (`lpp`), and pre-built `spatstat`
#' quadrature schemes (`quad`). Evaluates spatial covariates at quadrature
#' locations and constructs Berman-Turner weights \eqn{v_i} and dummy-process
#' intensities \eqn{\delta_i}.
#'
#' @param x A planar `ppp` object, a linear-network `lpp` object, or an existing
#'   `spatstat` quadrature scheme (`quad`).
#' @param covariates A named list of `im` objects (when `x` is a `ppp`) or
#'   `linim`/`linfun` objects (when `x` is an `lpp`) covering the observation
#'   domain. Numeric and factor/categorical covariates are supported. Ignored or
#'   treated as `X` fallback when `x` is a `quad` scheme.
#' @param nd Positive integer quadrature resolution parameter. For planar `ppp`,
#'   the number of dummy points along each coordinate axis (`nd x nd` grid, so
#'   values such as `32L`--`128L` are typical for 2D grids). For linear network
#'   `lpp`, determines equal spacing `eps = |L| / nd` across total network
#'   length `|L|` in [spatstat.linnet::linequad()]. Default is `1000L`.
#' @param quadrature_method Planar quadrature construction method: `"berman_turner"`
#'   (default, uses [spatstat.geom::quadscheme()]) or `"logistic"` (uses
#'   [spatstat.geom::quadscheme.logi()]). Automatically aligned when `objective`
#'   is supplied and `quadrature_method` is omitted.
#' @param dummytype Dummy point generation type for planar `ppp` data with
#'   `quadrature_method = "logistic"`: `"grid"` (default) or `"poisson"`.
#' @param objective Optional objective name (`"poisson"`, `"weighted_poisson"`,
#'   `"logistic"`, `"weighted_logistic"`, `"iterative_poisson"`, or
#'   `"iterative_logistic"`). Used to select planar `quadrature_method`
#'   automatically or extract logistic `dummy_intensity` from `quad` schemes
#'   (defaults to `"poisson"` when `x` is a `quad` scheme and `objective = NULL`).
#' @param X Covariate data matrix or data frame corresponding to all quadrature
#'   points when `x` is a pre-built `quad` scheme.
#' @param type Optional spatial domain type (`"planar"` or `"network"`) when `x`
#'   is a `quad` scheme; inferred automatically from `x` if omitted.
#' @param statistical_weight Optional numeric vector of custom statistical weights
#'   (\eqn{w_i}) matching the total number of quadrature points.
#' @param interaction_integral Optional numeric vector or scalar of spatial
#'   interaction integrals (\eqn{A_i}) for iteratively weighted objectives.
#' @param ... Additional arguments passed to [spatstat.geom::quadscheme()],
#'   [spatstat.geom::quadscheme.logi()], or [spatstat.linnet::linequad()].
#'
#' @details
#' `prepare_mlpp()` dispatches automatically based on the class of `x`:
#' - **Planar (`ppp`)**: Uses `"berman_turner"` (`nd x nd` cell weights \eqn{v_i}
#'   summing to \eqn{|W|}) or `"logistic"` quadrature recording \eqn{\delta_i}
#'   (`dummy_intensity`) as in Baddeley et al. (2014).
#' - **Linear network (`lpp`)**: Uses [spatstat.linnet::linequad()] with native
#'   network topology (`seg`, `tp`) preserved via `attr(Q, "situ")` so
#'   multi-level overpass crossings remain distinct. While Poisson weights
#'   \eqn{v_i} conserve total network length \eqn{|\mathcal{L}|} exactly, the
#'   homogeneous \eqn{\delta = m / |\mathcal{L}|} stored in `dummy_intensity`
#'   serves as a deterministic proxy for network logistic objectives.
#' - **Pre-built quadrature (`quad`)**: Converts any planar or network `spatstat`
#'   quadrature scheme into an `mlpp_data` object.
#'
#' Factor and character covariates are preserved in `data$X`. Downstream,
#' CatBoost models them natively via Ordered Target Statistics (`cat_features`),
#' while XGBoost and LightGBM consistently encode saved training factor levels
#' numerically.
#'
#' @return A validated `mlpp_data` object with `type` equal to `"planar"` or
#'   `"network"`.
#'
#' @references
#' Berman, M., & Turner, T. R. (1992). Approximating point process likelihoods
#' with GLIM. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*,
#' 41(1), 31--38. \doi{10.2307/2347614}
#'
#' Ang, Q. W., Baddeley, A., & Nair, G. (2012). Geometrically corrected second
#' order analysis of events on a linear network, with applications to ecology and
#' criminology. *Scandinavian Journal of Statistics*, 39(4), 591--617.
#' \doi{10.1111/j.1467-9469.2011.00762.x}
#'
#' Baddeley, A., Coeurjolly, J.-F., Rubak, E., & Waagepetersen, R. (2014).
#' Logistic regression for spatial Gibbs point processes. *Biometrika*,
#' 101(2), 377--392. \doi{10.1093/biomet/ast056}
#'
#' Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
#' Methodology and Applications with R*. Chapman and Hall/CRC.
#'
#' Baddeley, A., Nair, G., Rakshit, S., McSwiggan, G., & Davies, T. M. (2021).
#' Analysing point patterns on networks --- A review. *Spatial Statistics*,
#' 42, 100435. \doi{10.1016/j.spasta.2020.100435}
#'
#' @seealso [mlpp_data()], [attach_weights()], [attach_interaction()],
#'   [fit_gbpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE)) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(25, win = win)
#'   cov_x <- spatstat.geom::as.im(function(x, y) x, W = win)
#'   dat <- prepare_mlpp(pts, covariates = list(x = cov_x), nd = 16L)
#'   print(dat)
#'
#'   # Pre-built spatstat quadrature scheme
#'   Q <- spatstat.geom::quadscheme(pts, nd = 8L)
#'   X_q <- data.frame(x = spatstat.geom::coords(Q)$x)
#'   qdat <- prepare_mlpp(Q, X = X_q, objective = "poisson")
#'   print(qdat)
#' }
#'
#' if (requireNamespace("spatstat.data", quietly = TRUE) &&
#'   requireNamespace("spatstat.random", quietly = TRUE)) {
#'   data(simplenet, package = "spatstat.data")
#'   lpts <- spatstat.linnet::runiflpp(20, L = simplenet)
#'   lcov <- spatstat.linnet::linfun(function(x, y, seg, tp) x, L = simplenet)
#'   ldat <- prepare_mlpp(lpts, covariates = list(x = lcov), nd = 50L)
#'   print(ldat)
#' }
#' @export
prepare_mlpp <- function(
  x,
  covariates = list(),
  nd = 1000L,
  quadrature_method = c("berman_turner", "logistic"),
  dummytype = c("grid", "poisson"),
  objective = NULL,
  X = NULL,
  type = NULL,
  statistical_weight = NULL,
  interaction_integral = NULL,
  ...
) {
  if (inherits(x, "ppp")) {
    return(.prep_planar(
      point_pattern = x,
      covariates = covariates,
      nd = nd,
      quadrature_method = quadrature_method,
      dummytype = dummytype,
      missing_qm = missing(quadrature_method),
      objective = objective,
      statistical_weight = statistical_weight,
      interaction_integral = interaction_integral,
      ...
    ))
  }

  if (inherits(x, "lpp")) {
    return(.prep_net(
      point_pattern = x,
      covariates = covariates,
      nd = nd,
      statistical_weight = statistical_weight,
      interaction_integral = interaction_integral,
      ...
    ))
  }

  if (inherits(x, "quad")) {
    if (is.null(X) && (is.data.frame(covariates) || is.matrix(covariates))) {
      X <- covariates
    }
    if (is.null(objective)) {
      objective <- "poisson"
    }
    return(.prep_quad(
      Q = x,
      X = X,
      objective = objective,
      type = type,
      statistical_weight = statistical_weight,
      interaction_integral = interaction_integral
    ))
  }

  stop(
    "`x` must be a `spatstat` point pattern (`ppp` or `lpp`) or quadrature scheme (`quad`).",
    call. = FALSE
  )
}

#' @keywords internal
#' @noRd
.prep_planar <- function(
  point_pattern,
  covariates = list(),
  nd = 1000L,
  quadrature_method = c("berman_turner", "logistic"),
  dummytype = c("grid", "poisson"),
  missing_qm = FALSE,
  objective = NULL,
  statistical_weight = NULL,
  interaction_integral = NULL,
  ...
) {
  if (!is.list(covariates)) {
    stop("`covariates` must be a list of `im` objects.", call. = FALSE)
  }

  if (length(covariates) > 0L) {
    covariates <- lapply(covariates, function(z) {
      if (inherits(z, "imlist") && length(z) == 1L) z[[1L]] else z
    })

    if (is.null(names(covariates)) ||
      anyNA(names(covariates)) ||
      any(names(covariates) == "")) {
      stop("All planar covariates must be named.", call. = FALSE)
    }

    if (anyDuplicated(names(covariates))) {
      stop("Planar covariate names must be unique.", call. = FALSE)
    }

    invalid <- !vapply(covariates, inherits, logical(1), what = "im")
    if (any(invalid)) {
      stop("All planar covariates must be `im` objects.", call. = FALSE)
    }

    covariate_covers_window <- vapply(
      covariates,
      function(z) {
        spatstat.geom::is.subset.owin(
          point_pattern$window,
          spatstat.geom::Window(z)
        )
      },
      logical(1)
    )
    if (any(!covariate_covers_window)) {
      stop(
        "All planar covariates must cover the observation window of `point_pattern`.",
        call. = FALSE
      )
    }
  }

  if (length(nd) != 1L ||
    !is.numeric(nd) ||
    !is.finite(nd) ||
    nd <= 0 ||
    nd != as.integer(nd)) {
    stop("`nd` must be one positive integer.", call. = FALSE)
  }
  nd <- as.integer(nd)

  if (isTRUE(missing_qm) && !is.null(objective)) {
    logi_objs <- c("logistic", "weighted_logistic", "iterative_logistic")
    quadrature_method <- if (objective %in% logi_objs) {
      "logistic"
    } else {
      "berman_turner"
    }
  } else {
    quadrature_method <- match.arg(quadrature_method, c("berman_turner", "logistic"))
  }

  dummytype <- match.arg(dummytype, c("grid", "poisson"))

  if (quadrature_method == "berman_turner") {
    Q <- spatstat.geom::quadscheme(point_pattern, nd = nd, ...)
    U <- spatstat.geom::union.quad(Q)
    y <- as.integer(spatstat.geom::is.data(Q))
    v <- spatstat.geom::w.quad(Q)
    delta <- .dummy_intensity(y, measure = point_pattern$window)
    dummy_intensity <- rep(delta, length(y))
  } else {
    Q <- spatstat.geom::quadscheme.logi(
      point_pattern,
      dummytype = dummytype,
      nd = nd,
      ...
    )
    U <- spatstat.geom::union.quad(Q)
    y <- as.integer(spatstat.geom::is.data(Q))
    v <- spatstat.geom::w.quad(Q)
    dummy_intensity <- .dummy_intensity(Q)
  }

  X_df <- data.frame(
    row.names = seq_len(length(y)),
    check.names = FALSE
  )

  if (length(covariates) > 0L) {
    X_df <- as.data.frame(
      lapply(covariates, function(z) z[U]),
      check.names = FALSE
    )
    names(X_df) <- names(covariates)
  }

  if (nrow(X_df) != length(y)) {
    stop("Planar covariate evaluation produced an invalid number of rows.", call. = FALSE)
  }

  if (anyNA(X_df)) {
    stop("Planar covariates contain missing values at quadrature locations.", call. = FALSE)
  }

  locations <- as.data.frame(spatstat.geom::coords(U))

  mlpp_data(
    X = X_df,
    y = y,
    locations = locations,
    quadrature_weight = v,
    dummy_intensity = dummy_intensity,
    statistical_weight = statistical_weight,
    interaction_integral = interaction_integral,
    window = point_pattern$window,
    type = "planar",
    metadata = list(
      point_pattern = point_pattern,
      quadrature = Q,
      covariates = covariates,
      quadrature_method = quadrature_method,
      dummytype = if (quadrature_method == "logistic") dummytype else "grid"
    ),
    validate = TRUE
  )
}

#' @keywords internal
#' @noRd
.prep_net <- function(
  point_pattern,
  covariates = list(),
  nd = 1000L,
  statistical_weight = NULL,
  interaction_integral = NULL,
  ...
) {
  if (!is.list(covariates)) {
    stop("`covariates` must be a list of `linim` or `linfun` objects.", call. = FALSE)
  }

  if (length(covariates) > 0L) {
    if (is.null(names(covariates)) ||
      anyNA(names(covariates)) ||
      any(names(covariates) == "")) {
      stop("All network covariates must be named.", call. = FALSE)
    }

    if (anyDuplicated(names(covariates))) {
      stop("Network covariate names must be unique.", call. = FALSE)
    }

    invalid <- !vapply(
      covariates,
      function(z) inherits(z, "linim") || inherits(z, "linfun"),
      logical(1)
    )
    if (any(invalid)) {
      stop("All network covariates must be `linim` or `linfun` objects.", call. = FALSE)
    }

    network <- spatstat.linnet::as.linnet(point_pattern)
    .check_linnet(
      covariates,
      network,
      "All network covariates must be defined on the same linear network as `point_pattern`."
    )
  }

  if (length(nd) != 1L ||
    !is.numeric(nd) ||
    !is.finite(nd) ||
    nd <= 0 ||
    nd != as.integer(nd)) {
    stop("`nd` must be one positive integer.", call. = FALSE)
  }
  nd <- as.integer(nd)

  Q <- spatstat.linnet::linequad(point_pattern, nd = nd, ...)

  # Preserve native lpp object with (x, y, seg, tp) so overpass crossings remain distinct
  U <- attr(Q, "situ")
  if (is.null(U) || !inherits(U, "lpp")) {
    U <- spatstat.linnet::as.lpp(
      spatstat.geom::union.quad(Q),
      L = point_pattern$domain
    )
  }

  y <- as.integer(spatstat.geom::is.data(Q))
  v <- spatstat.geom::w.quad(Q)
  dummy_intensity <- .dummy_intensity(y, measure = v)

  X_df <- data.frame(
    row.names = seq_len(length(y)),
    check.names = FALSE
  )

  if (length(covariates) > 0L) {
    X_df <- as.data.frame(
      lapply(covariates, function(z) {
        if (inherits(z, "linfun")) {
          z(U$data$x, U$data$y, U$data$seg, U$data$tp)
        } else {
          z[U]
        }
      }),
      check.names = FALSE
    )
    names(X_df) <- names(covariates)
  }

  if (nrow(X_df) != length(y)) {
    stop("Network covariate evaluation produced an invalid number of rows.", call. = FALSE)
  }

  if (anyNA(X_df)) {
    stop("Network covariates contain missing values at quadrature locations.", call. = FALSE)
  }

  mlpp_data(
    X = X_df,
    y = y,
    locations = U,
    quadrature_weight = v,
    dummy_intensity = rep(dummy_intensity, length(y)),
    statistical_weight = statistical_weight,
    interaction_integral = interaction_integral,
    window = point_pattern$domain,
    type = "network",
    metadata = list(
      point_pattern = point_pattern,
      quadrature = Q,
      covariates = covariates,
      quadrature_method = "linequad"
    ),
    validate = TRUE
  )
}

#' @keywords internal
#' @noRd
.prep_quad <- function(
  Q,
  X,
  objective = c(
    "poisson",
    "weighted_poisson",
    "logistic",
    "weighted_logistic",
    "iterative_poisson",
    "iterative_logistic"
  ),
  type = NULL,
  statistical_weight = NULL,
  interaction_integral = NULL
) {
  objective <- match.arg(objective)
  situ <- attr(Q, "situ")

  if (is.null(type)) {
    type <- if ((!is.null(situ) && inherits(situ, "lpp")) || inherits(Q$data, "lpp")) {
      "network"
    } else {
      "planar"
    }
  } else {
    type <- match.arg(type, c("planar", "network"))
  }

  if (is.null(X) || (!is.data.frame(X) && !is.matrix(X))) {
    stop("`X` must be a data frame or matrix when preparing a `quad` scheme.", call. = FALSE)
  }

  y <- as.integer(spatstat.geom::is.data(Q))
  locations <- if (type == "network" && !is.null(situ) && inherits(situ, "lpp")) {
    situ
  } else {
    as.data.frame(spatstat.geom::coords(Q))
  }
  quadrature_weight <- spatstat.geom::w.quad(Q)

  if (nrow(X) != length(y)) {
    stop("`X` must have one row for each point in the quadrature scheme.", call. = FALSE)
  }

  if (length(quadrature_weight) != length(y)) {
    stop("`quadrature_weight` must have one value for each point.", call. = FALSE)
  }

  if (objective %in% c("weighted_poisson", "weighted_logistic") &&
    is.null(statistical_weight)) {
    stop("`statistical_weight` is required for weighted objectives.", call. = FALSE)
  }

  if (!is.null(statistical_weight) &&
    length(statistical_weight) != length(y)) {
    stop("`statistical_weight` must have one value for each point.", call. = FALSE)
  }

  dummy_intensity <- NULL
  if (objective %in% c("logistic", "weighted_logistic", "iterative_logistic")) {
    dummy_intensity <- .dummy_intensity(Q)
  }

  win_obj <- if (type == "network") {
    if (!is.null(situ) && inherits(situ, "lpp")) {
      situ$domain
    } else if (inherits(Q$data, "lpp")) {
      Q$data$domain
    } else {
      NULL
    }
  } else {
    if (!is.null(Q$data$window)) {
      Q$data$window
    } else if (inherits(Q$dummy, "ppp")) {
      Q$dummy$window
    } else {
      NULL
    }
  }

  mlpp_data(
    X = X,
    y = y,
    locations = locations,
    quadrature_weight = quadrature_weight,
    dummy_intensity = dummy_intensity,
    statistical_weight = statistical_weight,
    interaction_integral = interaction_integral,
    window = win_obj,
    type = type,
    metadata = list(
      objective = objective,
      source = "spatstat_quadrature",
      quadrature = Q
    ),
    validate = TRUE
  )
}
