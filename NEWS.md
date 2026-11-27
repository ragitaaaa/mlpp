# mlpp 0.1.0

* Initial release of **mlpp** (*Machine Learning for Point Processes*).
* Unified gradient boosting API (`fit_gbpp()`, `predict()`, `plot()`,
  `save_mlpp()`, `load_mlpp()`) interfacing with **XGBoost**,
  **LightGBM**, and **CatBoost** via `reticulate`, including TreeSHAP
  explainability (`explain_mlpp()`).
* Unified spatial quadrature preparation and data container (`prepare_mlpp()`,
  `mlpp_data()`) supporting planar point patterns (`ppp`), linear network
  point patterns (`lpp`), and pre-built `spatstat` quadrature schemes (`quad`).
* Six point process objectives: `"poisson"`, `"weighted_poisson"`,
  `"logistic"`, `"weighted_logistic"`, `"iterative_poisson"`, and
  `"iterative_logistic"`, with unified helpers `attach_weights()` and
  `attach_interaction()`.
* Independent stochastic thinning spatial cross-validation (`cv_mlpp()`,
  `plot.mlpp_cv()`) and unified Poisson evaluation (`eval_mlpp()`).
* Python environment setup and diagnostics (`install_mlpp_python()`,
  `check_mlpp_python()`).
