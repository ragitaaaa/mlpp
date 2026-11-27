#' Estimate and attach a spatial interaction integral to mlpp data
#'
#' Estimates the spatial interaction integral \eqn{A_i} used by the iteratively
#' weighted objectives (`"iterative_poisson"`, `"iterative_logistic"`) on either
#' a planar domain (`type = "planar"`) or linear network (`type = "network"`) and
#' attaches the resulting vector to `data$interaction_integral`.
#'
#' @param data An [mlpp_data] object.
#' @param rmax Single positive finite numeric value giving the maximum distance
#'   included in the interaction integral.
#' @param point_pattern Optional observed point pattern (`ppp` for planar data,
#'   `lpp` for network data). If `NULL` (default), automatically uses
#'   `data$metadata$point_pattern` stored by [prepare_mlpp()].
#' @param lambda Optional intensity specification passed to
#'   [spatstat.explore::Kinhom()] (planar) or
#'   [spatstat.linnet::linearpcfinhom()] (network).
#' @param correction Edge or network geometry correction. Defaults to `"translate"`
#'   for planar patterns (choices: `"translate"`, `"isotropic"`, `"border"`;
#'   `"isotropic"` falls back to `"translate"` on pixel-mask windows) and
#'   `"Ang"` for linear networks (choices: `"Ang"`, `"none"`).
#' @param nr Integer number of distance grid points (\eqn{\ge 2}) in `[0, rmax]`
#'   for network pair-correlation estimation (`data$type == "network"` only).
#'   Default is `128L`.
#' @param exact Logical indicating whether network integration (`data$type == "network"`
#'   only) uses exact segment-wise integration
#'   ([spatstat.linnet::integral.linfun()]). Default is `TRUE`.
#' @param nd Number of network integration points when `data$type == "network"`
#'   and `exact = FALSE`. Default is `1000L`.
#'
#' @details
#' For planar data (`data$type == "planar"`), estimates the inhomogeneous Ripley's
#' K-function via [spatstat.explore::Kinhom()] at distance `rmax` and computes
#' \deqn{A = \hat{K}_{\mathrm{inhom}}(r_{\max}) - \pi r_{\max}^2,}
#' recycled across all quadrature points in `data`.
#'
#' For linear network data (`data$type == "network"`), estimates the pair
#' correlation \eqn{\hat{g}(r)} on `[0, rmax]` using [spatstat.linnet::linearpcf()]
#' when `lambda = NULL` or [spatstat.linnet::linearpcfinhom()] when `lambda` is
#' provided, sets \eqn{g(r) = 1} for \eqn{r > r_{\max}}, and evaluates the
#' location-specific integral at every quadrature point \eqn{u \in \mathcal{L}}:
#' \deqn{A(u) = \int_{\mathcal{L}} [g(d_{\mathcal{L}}(u, s)) - 1]\, d\ell(s).}
#'
#' @return The modified [mlpp_data] object with `interaction_integral` attached.
#'
#' @references
#' Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for
#' inhomogeneous spatial point processes. *Biometrika*, 97(4), 867--880.
#' \doi{10.1093/biomet/asq047}
#'
#' Ang, Q. W., Baddeley, A., & Nair, G. (2012). Geometrically corrected second
#' order analysis of events on a linear network, with applications to ecology and
#' criminology. *Scandinavian Journal of Statistics*, 39(4), 591--617.
#' \doi{10.1111/j.1467-9469.2011.00762.x}
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
#' @seealso [prepare_mlpp()], [attach_weights()], [fit_gbpp()]
#'
#' @examples
#' if (requireNamespace("spatstat.random", quietly = TRUE)) {
#'   win <- spatstat.geom::square(1)
#'   pts <- spatstat.random::runifpoint(30, win = win)
#'   dat <- prepare_mlpp(pts, nd = 16L)
#'   dat <- attach_interaction(dat, rmax = 0.15)
#'   head(dat$interaction_integral)
#' }
#'
#' if (requireNamespace("spatstat.data", quietly = TRUE) &&
#'   requireNamespace("spatstat.random", quietly = TRUE)) {
#'   data(simplenet, package = "spatstat.data")
#'   lpts <- spatstat.linnet::runiflpp(15, L = simplenet)
#'   ldat <- prepare_mlpp(lpts, nd = 12L)
#'   ldat <- attach_interaction(ldat, rmax = 0.08, nr = 16L, exact = FALSE, nd = 20L)
#'   head(ldat$interaction_integral)
#' }
#' @export
attach_interaction <- function(
  data,
  rmax,
  point_pattern = NULL,
  lambda = NULL,
  correction = NULL,
  nr = 128L,
  exact = TRUE,
  nd = 1000L
) {
  if (!inherits(data, "mlpp_data")) {
    stop("`data` must be an `mlpp_data` object.", call. = FALSE)
  }

  if (is.null(point_pattern)) {
    point_pattern <- data$metadata$point_pattern
    if (is.null(point_pattern)) {
      stop(
        "`point_pattern` is required when `data$metadata$point_pattern` is absent.",
        call. = FALSE
      )
    }
  }

  if (identical(data$type, "planar")) {
    if (!inherits(point_pattern, "ppp")) {
      stop("`point_pattern` must be a `ppp` object for planar data.", call. = FALSE)
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

    A_val <- .est_interaction(
      x = point_pattern,
      rmax = rmax,
      lambda = lambda,
      correction = correction
    )
    data$interaction_integral <- rep(A_val, length.out = length(data$y))
    .check_data(data)
    return(data)
  }

  if (!inherits(point_pattern, "lpp")) {
    stop("`point_pattern` must be an `lpp` object for network data.", call. = FALSE)
  }

  quadrature_locations <- NULL
  if (!is.null(data$metadata) && !is.null(data$metadata$quadrature)) {
    quadrature_locations <- attr(data$metadata$quadrature, "situ")
  }
  if (!inherits(quadrature_locations, "lpp") && inherits(data$locations, "lpp")) {
    quadrature_locations <- data$locations
  }
  if (!inherits(quadrature_locations, "lpp")) {
    stop("Network quadrature locations (`lpp`) are unavailable in `data`.", call. = FALSE)
  }

  if (!identical(
    spatstat.linnet::as.linnet(point_pattern),
    spatstat.linnet::as.linnet(quadrature_locations)
  )) {
    stop(
      "`point_pattern` and the network quadrature must be defined on the same linear network.",
      call. = FALSE
    )
  }

  if (spatstat.geom::npoints(quadrature_locations) != nrow(data$X)) {
    stop("Network quadrature locations must match the rows of `data$X`.", call. = FALSE)
  }

  A_vec <- .est_interaction(
    x = point_pattern,
    rmax = rmax,
    locations = quadrature_locations,
    lambda = lambda,
    correction = correction,
    nr = nr,
    exact = exact,
    nd = nd
  )

  if (length(A_vec) != nrow(data$X)) {
    stop("Estimated interaction integral has an invalid length.", call. = FALSE)
  }

  data$interaction_integral <- A_vec
  .check_data(data)
  data
}

#' Estimate planar or network spatial interaction integral
#'
#' @keywords internal
#' @noRd
.est_interaction <- function(
  x,
  rmax,
  locations = NULL,
  lambda = NULL,
  correction = NULL,
  nr = 128L,
  exact = TRUE,
  nd = 1000L
) {
  if (length(rmax) != 1L || !is.numeric(rmax) || !is.finite(rmax) || rmax <= 0) {
    stop("`rmax` must be a single positive finite numeric value.", call. = FALSE)
  }

  if (inherits(x, "ppp")) {
    if (is.null(correction)) correction <- "translate"
    correction <- match.arg(correction, c("translate", "isotropic", "border"))
    if (correction == "isotropic" && spatstat.geom::is.mask(spatstat.geom::Window(x))) {
      correction <- "translate"
    }

    if (!is.null(lambda)) {
      if (!is.numeric(lambda)) {
        stop("`lambda` must be numeric.", call. = FALSE)
      }
      if (length(lambda) != spatstat.geom::npoints(x)) {
        stop("`lambda` must contain one value for each point in `x`.", call. = FALSE)
      }
      if (any(!is.finite(lambda)) || any(lambda <= 0)) {
        stop("`lambda` must contain positive finite values.", call. = FALSE)
      }
    }

    r_values <- c(0, as.numeric(rmax))
    K <- if (is.null(lambda)) {
      spatstat.explore::Kinhom(x, r = r_values, correction = correction)
    } else {
      spatstat.explore::Kinhom(x, lambda = lambda, r = r_values, correction = correction)
    }

    K_data <- as.data.frame(K)
    col_nm <- switch(correction,
      isotropic = "iso",
      translate = "trans",
      border = "border"
    )
    if (!col_nm %in% names(K_data)) {
      stop(
        paste0("The requested K-function correction `", correction, "` was not returned by `Kinhom()`."),
        call. = FALSE
      )
    }

    K_val <- K_data[[col_nm]][which.min(abs(K_data$r - rmax))]
    if (length(K_val) != 1L || !is.finite(K_val)) {
      stop("The K-function estimate at `rmax` is not finite.", call. = FALSE)
    }

    out <- K_val - pi * rmax^2
    if (!is.finite(out)) {
      stop("The interaction integral is not finite.", call. = FALSE)
    }
    return(out)
  }

  if (!inherits(x, "lpp")) {
    stop("`x` must be a `ppp` or `lpp` object.", call. = FALSE)
  }

  if (is.null(locations)) locations <- x
  if (!inherits(locations, "lpp")) {
    stop("`locations` must be an `lpp` object.", call. = FALSE)
  }
  if (is.null(correction)) correction <- "Ang"
  if (!is.character(correction) || length(correction) != 1L || !correction %in% c("Ang", "none")) {
    stop("`correction` must be either \"Ang\" or \"none\".", call. = FALSE)
  }
  if (length(nr) != 1L || !is.numeric(nr) || !is.finite(nr) || nr < 2 || nr != as.integer(nr)) {
    stop("`nr` must be one integer greater than or equal to 2.", call. = FALSE)
  }
  if (length(exact) != 1L || !is.logical(exact) || is.na(exact)) {
    stop("`exact` must be one logical value.", call. = FALSE)
  }
  if (length(nd) != 1L || !is.numeric(nd) || !is.finite(nd) || nd <= 0 || nd != as.integer(nd)) {
    stop("`nd` must be one positive integer.", call. = FALSE)
  }

  nr <- as.integer(nr)
  nd <- as.integer(nd)
  network <- spatstat.linnet::as.linnet(x)
  if (!identical(spatstat.linnet::as.linnet(locations), network)) {
    stop("`x` and `locations` must be defined on the same linear network.", call. = FALSE)
  }

  if (spatstat.geom::npoints(locations) == 0L) {
    return(numeric(0))
  }

  r_grid <- seq(from = 0, to = rmax, length.out = nr)
  g_hat <- if (is.null(lambda)) {
    spatstat.linnet::linearpcf(X = x, r = r_grid, correction = correction)
  } else {
    spatstat.linnet::linearpcfinhom(X = x, lambda = lambda, r = r_grid, correction = correction)
  }

  g_values <- g_hat$est
  if (!is.numeric(g_values) || anyNA(g_values) || any(!is.finite(g_values))) {
    stop("Estimated network pair correlation contains non-finite values.", call. = FALSE)
  }

  g_fn <- as.function(g_hat, value = "est", extrapolate = FALSE)
  pair_correlation <- function(distance) {
    value <- rep(1, length(distance))
    inside <- distance <= rmax
    if (any(inside)) {
      est <- g_fn(distance[inside])
      if (anyNA(est) || any(!is.finite(est))) {
        stop("Network pair-correlation interpolation produced non-finite values.", call. = FALSE)
      }
      value[inside] <- est
    }
    value
  }

  .net_integral(
    locations = locations,
    network = network,
    pair_correlation = pair_correlation,
    exact = exact,
    nd = nd
  )
}

#' Evaluate linear-network interaction integral for a given pair correlation
#'
#' @keywords internal
#' @noRd
.net_integral <- function(
  locations,
  network,
  pair_correlation,
  exact = FALSE,
  nd = 1000L
) {
  if (!inherits(locations, "lpp")) {
    stop("`locations` must be an `lpp` object.", call. = FALSE)
  }
  if (!inherits(network, "linnet")) {
    stop("`network` must be a `linnet` object.", call. = FALSE)
  }
  if (!is.function(pair_correlation)) {
    stop("`pair_correlation` must be a function.", call. = FALSE)
  }
  if (length(exact) != 1L || is.na(exact) || !is.logical(exact)) {
    stop("`exact` must be one logical value.", call. = FALSE)
  }
  if (length(nd) != 1L || !is.numeric(nd) || !is.finite(nd) || nd <= 0 || nd != as.integer(nd)) {
    stop("`nd` must be one positive integer.", call. = FALSE)
  }

  nd <- as.integer(nd)
  if (!identical(spatstat.linnet::as.linnet(locations), network)) {
    stop("`locations` and `network` must be defined on the same linear network.", call. = FALSE)
  }

  n_locations <- spatstat.geom::npoints(locations)
  if (n_locations == 0L) {
    return(numeric(0))
  }

  result <- numeric(n_locations)
  for (i in seq_len(n_locations)) {
    distance_fn <- spatstat.geom::distfun(locations[i])

    integrand <- spatstat.linnet::linfun(
      function(x, y, seg, tp) {
        distance <- distance_fn(x, y, seg = seg, tp = tp)
        value <- pair_correlation(distance)
        if (length(value) != length(distance)) {
          stop("`pair_correlation` must return one value for each distance.", call. = FALSE)
        }
        if (!is.numeric(value) || anyNA(value) || any(!is.finite(value))) {
          stop("`pair_correlation` must return finite numeric values.", call. = FALSE)
        }
        value - 1
      },
      L = network
    )

    result[i] <- spatstat.linnet::integral.linfun(integrand, exact = exact, nd = nd)
  }

  if (anyNA(result) || any(!is.finite(result))) {
    stop("Network interaction integral produced non-finite values.", call. = FALSE)
  }

  result
}
