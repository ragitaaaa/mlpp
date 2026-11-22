# mlpp: Machine Learning for Point Processes on Planar and Linear Network Domains

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![R-CMD-check](https://github.com/ragitaaaa/mlpp/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ragitaaaa/mlpp/actions/workflows/R-CMD-check.yaml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

`mlpp` is an R package for fitting machine learning models—currently featuring gradient boosted decision trees (GBDT)—to spatial point process data. It integrates spatial point process methodology with gradient boosting backends (**LightGBM**, **CatBoost**, and **XGBoost**) via `reticulate`.

The package supports both two-dimensional planar Euclidean windows ($W \subset \mathbb{R}^2$) and one-dimensional linear networks ($\mathcal{L} \subset \mathbb{R}^2$, such as street or river networks), alongside stochastic thinning spatial cross-validation, TreeSHAP model explanations, and session-safe model serialization.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation & Python Setup](#installation--python-setup)
  - [Step 1: Install R Package](#step-1-install-r-package)
  - [Step 2: Automated Python Environment Configuration](#step-2-automated-python-environment-configuration)
  - [Step 3: Verification & Diagnostics](#step-3-verification--diagnostics)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [Workflow 1: 2D Planar Point Processes](#workflow-1-2d-planar-point-processes)
  - [Workflow 2: 1D Linear Road Networks](#workflow-2-1d-linear-road-networks)
  - [Workflow 3: Spatial Cross-Validation via Stochastic Thinning](#workflow-3-spatial-cross-validation-via-stochastic-thinning)
  - [Workflow 4: Model Interpretability (TreeSHAP & Partial Dependence)](#workflow-4-model-interpretability-treeshap--partial-dependence)
- [API Reference Summary](#api-reference-summary)
- [References](#references)
- [License](#license)

---

## Key Features

- **Gradient Boosting Backends:** Unified R interface for **LightGBM**, **CatBoost**, and **XGBoost** with native categorical covariate support.
- **Spatial Domains:** Support for planar point patterns (`ppp` from `spatstat.geom`) and linear network point patterns (`lpp` from `spatstat.linnet`).
- **Point Process Likelihood Objectives:**
  - **Poisson loss:** Standard (`"poisson"`), statically weighted (`"weighted_poisson"`), and iteratively weighted (`"iterative_poisson"`; Lu et al., 2025) Poisson composite likelihoods.
  - **Logistic loss:** Standard (`"logistic"`), statically weighted (`"weighted_logistic"`), and iteratively weighted (`"iterative_logistic"`; Choiruddin et al., 2026) logistic likelihoods.
- **Validation & Diagnostics:** Independent stochastic thinning spatial cross-validation (`cv_gbpp()`) with conserved domain measure, alongside `calc_poisson_loglik()` and `calc_poisson_deviance()`.
- **Model Interpretation:** TreeSHAP feature attributions and dependence plots (`explain_gbpp()`) on the additive log-intensity (link) scale.
- **Spatial Predictions & Persistence:** Continuous spatial intensity surfaces as `spatstat` pixel images (`im` and `linim`), plus cross-session model saving/loading (`save_gbpp_model()`, `load_gbpp_model()`).

---

## Installation & Python Setup

`mlpp` interfaces with LightGBM, CatBoost, and XGBoost via `reticulate`.

### Step 1: Install R Package

Ensure you have R (>= 4.0.0) installed. Install `mlpp` from GitHub:

```r
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("ragitaaaa/mlpp")
```

To update an existing installation from GitHub, restart your R session and re-run `remotes::install_github("ragitaaaa/mlpp", force = TRUE, upgrade = "never")`.

### Step 2: Automated Python Environment Configuration

Run `install_gbpp_python()` in R to create an isolated virtual environment (`"r-gbpp"`) containing `numpy`, `scipy`, `pandas`, `xgboost`, `lightgbm`, and `catboost`:

```r
library(mlpp)

# Automated setup (creates isolated virtualenv 'r-gbpp')
install_gbpp_python()
```

#### Custom or Pre-Existing Python Environments

If you already maintain a virtual environment or Conda environment with the required Python libraries, point `mlpp` to it before fitting models:

```r
# Option A: Set environment variable
Sys.setenv(GBPP_PYTHON = "/path/to/your/python")

# Option B: Use reticulate directly before initializing Python
reticulate::use_virtualenv("my-spatial-env", required = TRUE)
```

### Step 3: Verification & Diagnostics

```r
library(mlpp)

# Check active Python configuration and installed engine versions
check_gbpp_python()
```

---

## Step-by-Step Usage Guide

### Workflow 1: 2D Planar Point Processes

This example models the spatial distribution of *Beilschmiedia pendula* trees from the `bei` dataset on Barro Colorado Island using elevation and slope gradient covariates.

#### 1. Load Point Pattern and Covariates

```r
library(mlpp)
library(spatstat.geom)
library(spatstat.data)

# data(bei) loads both 'bei' (ppp) and 'bei.extra' (list of im covariates)
data(bei, package = "spatstat.data")
```

#### 2. Prepare Quadrature Scheme (`prepare_gbpp`)

```r
planar_data <- prepare_gbpp(
  point_pattern = bei,
  covariates = bei.extra,
  nd = 128L,
  quadrature_method = "logistic"
)

print(planar_data)
```

#### 3. Fit Model (`fit_gbpp`)

```r
fit_planar <- fit_gbpp(
  data = planar_data,
  engine = "lightgbm",
  objective = "logistic",
  n_trees = 150L,
  learning_rate = 0.05,
  max_depth = 5L,
  subsample = 0.8,
  colsample_bytree = 0.8,
  random_seed = 42L
)

print(fit_planar)
```

#### 4. Predict Intensity Surface & Evaluate Fit Metrics

```r
# Continuous intensity pixel image (im)
pred_im <- predict(fit_planar, newdata = bei.extra, type = "im")
plot(pred_im, main = "Estimated Intensity: Beilschmiedia pendula (LightGBM)")
plot(bei, add = TRUE, pch = ".", col = rgb(0, 0, 0, 0.25))

# Quadrature evaluation (log-likelihood, deviance, and total abundance check)
pred_int <- predict(fit_planar, newdata = planar_data$X, type = "intensity")
calc_poisson_loglik(pred_int, planar_data$y, planar_data$quadrature_weight)
calc_poisson_deviance(pred_int, planar_data$y, planar_data$quadrature_weight)
c(
  expected = sum(planar_data$quadrature_weight * pred_int),
  actual = sum(planar_data$y == 1)
)
```

---

### Workflow 2: 1D Linear Road Networks

`mlpp` supports network-constrained point processes via `spatstat.linnet`.

#### 1. Load Linear Network Point Pattern

```r
library(mlpp)
library(spatstat.geom)
library(spatstat.linnet)
library(spatstat.data)

data(chicago, package = "spatstat.data")
net <- as.linnet(chicago)
net_covs <- list(
  x_coord = as.linim(linfun(function(x, y, seg, tp) x, net)),
  y_coord = as.linim(linfun(function(x, y, seg, tp) y, net))
)
```

#### 2. Prepare Network Quadrature (`prepare_gbpp_network`)

```r
net_data <- prepare_gbpp_network(
  point_pattern = chicago,
  covariates = net_covs,
  nd = 100L
)

print(net_data)
```

#### 3. Fit Model with CatBoost Backend

```r
fit_net <- fit_gbpp(
  data = net_data,
  engine = "catboost",
  objective = "poisson",
  n_trees = 100L,
  learning_rate = 0.04,
  max_depth = 5L
)

print(fit_net)
```

#### 4. Predict Network Intensity Surface (`linim`)

```r
pred_linim <- predict(fit_net, newdata = net_covs, type = "linim")
plot(pred_linim, main = "Estimated Street Crime Intensity on Chicago Network")
plot(as.ppp(chicago), add = TRUE, pch = 16, cex = 0.4, col = "red")
```

---

### Workflow 3: Spatial Cross-Validation via Stochastic Thinning

`cv_gbpp()` performs independent $V$-fold stochastic thinning cross-validation (Cronie et al., 2024), recomputing fold quadrature weights so domain area $|W|$ or network length $|\mathcal{L}|$ is conserved.

```r
cv_results <- cv_gbpp(
  data = planar_data,
  engine = "lightgbm",
  objective = "logistic",
  folds = 5L,
  n_trees = 100L,
  seed = 42L,
  params = list(
    learning_rate = 0.05,
    max_depth = 5L
  )
)

print(cv_results)
plot(cv_results)
```

---

### Workflow 4: Model Interpretability (TreeSHAP & Partial Dependence)

`explain_gbpp()` extracts exact TreeSHAP feature attributions on the additive log-intensity (link) scale against the fitted trees.

```r
# Extract TreeSHAP values over dummy quadrature rows
shap_output <- explain_gbpp(
  object = fit_planar,
  data = planar_data,
  background = "quadrature"
)

print(shap_output)
plot(shap_output, type = "importance")
plot(shap_output, type = "dependence", feature = "elev")
plot(shap_output, type = "dependence", feature = "grad")
```

---

## API Reference Summary

| Function | Primary Purpose | Key Arguments |
|---|---|---|
| `install_gbpp_python()` | Automated Python virtualenv setup | `envname = "r-gbpp"`, `method = "auto"` |
| `check_gbpp_python()` | Inspect Python config and engine versions | *(None)* |
| `prepare_gbpp()` | Construct 2D planar quadrature data | `point_pattern`, `covariates`, `nd`, `quadrature_method` |
| `prepare_gbpp_network()` | Construct 1D linear network quadrature data | `point_pattern`, `covariates`, `nd` |
| `attach_stat_weight()` | Attach custom weights to `gbpp_data` | `data`, `weights` |
| `attach_interaction()` | Attach planar K interaction integral for iterative objectives | `data`, `point_pattern`, `rmax`, `correction` |
| `attach_net_interaction()` | Attach network interaction integral for iterative objectives | `data`, `point_pattern`, `rmax`, `correction` |
| `fit_gbpp()` | Fit gradient boosted point process model | `data`, `engine`, `objective`, `n_trees`, `learning_rate`, `max_depth`, `baseline` |
| `predict.gbpp_model()` | Predict link, intensity, raw margin, `im`, or `linim` | `object`, `newdata`, `type = c("link", "intensity", "raw", "im", "linim")` |
| `plot.gbpp_model()` | Plot continuous spatial intensity or link surface | `x`, `newdata`, `type = c("intensity", "link")`, `overlay_points` |
| `save_gbpp_model()` / `load_gbpp_model()` | Save and restore model bundles across R sessions | `object`, `file` |
| `calc_poisson_loglik()` | Evaluate spatial Poisson log-likelihood | `pred_intensity`, `y`, `w`, `p_scale` |
| `calc_poisson_deviance()` | Evaluate spatial Poisson deviance | `pred_intensity`, `y`, `w`, `p_scale` |
| `cv_gbpp()` | Spatial stochastic thinning cross-validation | `data`, `engine`, `params`, `n_trees`, `folds`, `objective`, `seed` |
| `explain_gbpp()` | TreeSHAP feature attributions on link scale | `object`, `data`, `background` |
| `plot.gbpp_explanation()` | Visualize TreeSHAP importance or dependence plots | `x`, `type = c("importance", "dependence")`, `feature` |
| `plot.gbpp_cv()` | Visualize stochastic thinning validation folds | `x`, `separate` |

---

## References

- Ang, Q. W., Baddeley, A., & Nair, G. (2012). Geometrically corrected second order analysis of events on a linear network, with applications to ecology and criminology. *Scandinavian Journal of Statistics*, 39(4), 591–617. doi:10.1111/j.1467-9469.2011.00762.x
- Baddeley, A., Coeurjolly, J.-F., Rubak, E., & Waagepetersen, R. (2014). Logistic regression for spatial Gibbs point processes. *Biometrika*, 101(2), 377–392. doi:10.1093/biomet/ast056
- Berman, M., & Turner, T. R. (1992). Approximating point process likelihoods with GLIM. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 41(1), 31–38. doi:10.2307/2347614
- Choiruddin, A., Tasbihi, N. L., Abbad, M. A., Syukron, M., Widhianingsih, T. D. A., & Mateu, J. (2026). XGBoost algorithm for a spatial point process intensity estimation based on logistic loss. *Computational Journal of Mathematical and Statistical Sciences*, 5(2), 472–496. doi:10.21608/cjmss.2026.452469.1353
- Cronie, O., Moradi, M., & Biscio, C. A. N. (2024). A cross-validation-based statistical theory for point processes. *Biometrika*, 111(2), 625–641. doi:10.1093/biomet/asad041
- Guan, Y., & Shen, Y. (2010). A weighted estimating equation approach for inhomogeneous spatial point processes. *Biometrika*, 97(4), 867–880. doi:10.1093/biomet/asq047
- Lu, C., Guan, Y., van Lieshout, M.-C., & Xu, G. (2025). XGBoostPP: Tree-based estimation of point process intensity functions. *Journal of Computational and Graphical Statistics*, 35(1), 229–240. doi:10.1080/10618600.2025.2520582
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67. doi:10.1038/s42256-019-0138-9

---

## License

This package is open-source software licensed under the [MIT License](LICENSE).
