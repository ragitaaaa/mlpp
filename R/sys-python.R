#' Configure and validate the Python environment used by mlpp
#'
#' Configures `reticulate` to use the Python interpreter specified by
#' `MLPP_PYTHON` (with fallback to `GBPP_PYTHON` or `RETICULATE_PYTHON`) or,
#' when none is set, auto-discovers a project-local `.venv` or existing
#' `"r-mlpp"` / `"r-gbpp"` environment.
#'
#' @return Invisibly returns `TRUE` if the Python environment is available.
#' @keywords internal
#' @noRd
.init_python <- function() {
  if (reticulate::py_available(initialize = FALSE)) {
    return(invisible(TRUE))
  }

  python <- Sys.getenv("MLPP_PYTHON")
  if (!nzchar(python)) {
    python <- Sys.getenv("GBPP_PYTHON")
  }
  if (!nzchar(python)) {
    python <- Sys.getenv("RETICULATE_PYTHON")
  }

  if (!nzchar(python)) {
    curr <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
    candidate <- NULL

    for (i in seq_len(5L)) {
      win_py <- file.path(curr, ".venv", "Scripts", "python.exe")
      unix_py <- file.path(curr, ".venv", "bin", "python")

      if (file.exists(win_py)) {
        candidate <- win_py
        break
      } else if (file.exists(unix_py)) {
        candidate <- unix_py
        break
      }

      parent <- dirname(curr)
      if (identical(parent, curr)) break
      curr <- parent
    }

    if (is.null(candidate) || !file.exists(candidate)) {
      for (env in c("r-mlpp", "r-gbpp")) {
        if (tryCatch(reticulate::virtualenv_exists(env), error = function(e) FALSE)) {
          reticulate::use_virtualenv(env, required = TRUE)
          return(invisible(TRUE))
        }
        if (tryCatch(reticulate::condaenv_exists(env), error = function(e) FALSE)) {
          reticulate::use_condaenv(env, required = TRUE)
          return(invisible(TRUE))
        }
      }
      disc <- tryCatch(reticulate::py_discover_config(), error = function(e) NULL)
      if (!is.null(disc) && nzchar(disc$python) && file.exists(disc$python)) {
        candidate <- disc$python
      }
    }

    if (!is.null(candidate) && file.exists(candidate)) {
      python <- candidate
    } else {
      stop(
        paste0(
          "MLPP Python environment could not be found.\n",
          "You can automatically set up a dedicated environment with all required ",
          "packages by running:\n",
          "  mlpp::install_mlpp_python()\n",
          "Or configure an existing Python interpreter via ",
          "`Sys.setenv(MLPP_PYTHON = 'path/to/python')`."
        ),
        call. = FALSE
      )
    }
  }

  if (!file.exists(python)) {
    stop(
      paste0(
        "The Python executable does not exist: ",
        python
      ),
      call. = FALSE
    )
  }

  reticulate::use_python(
    python,
    required = TRUE
  )

  invisible(TRUE)
}

#' Locate mlpp's bundled Python backend files
#'
#' Resolves the filesystem path to the package's `inst/python` directory containing
#' custom point process objectives and engine adapters.
#'
#' @return Character scalar containing the normalized path to the Python backend
#'   module directory.
#' @keywords internal
#' @noRd
.py_path <- function() {
  path <- system.file(
    "python",
    package = "mlpp"
  )

  if (nzchar(path) && dir.exists(path)) {
    return(normalizePath(path, winslash = "/"))
  }

  curr <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  for (i in seq_len(5L)) {
    candidate <- file.path(curr, "inst", "python")
    if (dir.exists(candidate)) {
      return(normalizePath(candidate, winslash = "/"))
    }
    parent <- dirname(curr)
    if (identical(parent, curr)) break
    curr <- parent
  }

  stop(
    "MLPP Python backend files could not be found.",
    call. = FALSE
  )
}

#' Install Python dependencies for mlpp
#'
#' Automatically creates an isolated virtual environment (`"r-mlpp"`) and
#' installs the required Python dependencies: `numpy`, `scipy`, `pandas`,
#' `xgboost`, `lightgbm`, and `catboost`.
#'
#' @param envname Name of the Python virtual environment. Default is `"r-mlpp"`.
#' @param method Method passed to [reticulate::py_install()], e.g. `"auto"`,
#'   `"virtualenv"`, or `"conda"`.
#' @param ... Additional arguments passed to [reticulate::py_install()].
#'
#' @details
#' Installs the Python packages needed by [fit_gbpp()] and [explain_mlpp()].
#' Calling this function downloads packages and modifies or creates a Python
#' environment; in automated checks it should not be run unconditionally.
#'
#' @return Invisibly returns `TRUE` on success.
#'
#' @references
#' Ushey, K., Allaire, J. J., & Tang, Y. (2024). *reticulate: Interface to Python*.
#' R package.
#'
#' @seealso [check_mlpp_python()], [fit_gbpp()]
#'
#' @examples
#' \dontrun{
#' install_mlpp_python(envname = "r-mlpp")
#' }
#' @export
install_mlpp_python <- function(
  envname = "r-mlpp",
  method = "auto",
  ...
) {
  pkgs <- c("numpy", "scipy", "pandas", "xgboost", "lightgbm", "catboost")
  message(
    "Installing MLPP Python dependencies (",
    paste(pkgs, collapse = ", "),
    ") into environment '", envname, "'..."
  )
  reticulate::py_install(
    packages = pkgs,
    envname = envname,
    method = method,
    ...
  )
  message("MLPP Python dependencies installed successfully!")
  invisible(TRUE)
}

#' Check mlpp Python Environment and Engine Status
#'
#' Verifies that Python is properly configured and inspects the availability and
#' versions of required packages (`lightgbm`, `catboost`, `xgboost`, `numpy`,
#' `scipy`, `pandas`).
#'
#' @details
#' Discovers Python using `MLPP_PYTHON` (or `GBPP_PYTHON` / `RETICULATE_PYTHON`),
#' any nearby project `.venv`, or an existing `"r-mlpp"` / `"r-gbpp"` environment,
#' then prints the active Python executable, version, and module availability table.
#'
#' @return Invisibly returns a named logical vector of package availability
#'   (`TRUE` when importable, `FALSE` otherwise).
#'
#' @references
#' Ushey, K., Allaire, J. J., & Tang, Y. (2024). *reticulate: Interface to Python*.
#' R package.
#'
#' @seealso [install_mlpp_python()], [fit_gbpp()]
#'
#' @examples
#' if (reticulate::py_available()) {
#'   status <- check_mlpp_python()
#'   print(status)
#' }
#' @export
check_mlpp_python <- function() {
  cat("MLPP Python Environment Diagnostic\n")
  cat("------------------------------------\n")
  .init_python()
  py_config <- reticulate::py_config()
  cat("Python executable :", py_config$python, "\n")
  cat("Python version    :", as.character(py_config$version), "\n")

  pkgs <- c("numpy", "scipy", "pandas", "lightgbm", "catboost", "xgboost")
  status <- logical(length(pkgs))
  names(status) <- pkgs

  cat("\nRequired Boosting Packages:\n")
  for (pkg in pkgs) {
    avail <- reticulate::py_module_available(pkg)
    status[pkg] <- avail
    if (avail) {
      mod <- tryCatch(reticulate::import(pkg), error = function(e) NULL)
      ver <- if (!is.null(mod) && !is.null(mod$`__version__`)) mod$`__version__` else "available"
      cat(sprintf("  %-10s : [OK] version %s\n", pkg, ver))
    } else {
      cat(sprintf("  %-10s : [NOT FOUND]\n", pkg))
    }
  }
  invisible(status)
}
