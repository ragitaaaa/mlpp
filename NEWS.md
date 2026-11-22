# mlpp 0.1.0

* Initial release of **mlpp** (*Machine Learning for Point Processes*).
* Unified gradient boosting API (`fit_gbpp()`, `predict()`, `plot()`,
  `save_gbpp_model()`, `load_gbpp_model()`) interfacing with **XGBoost**,
  **LightGBM**, and **CatBoost** via `reticulate`, including native categorical
  feature handling and TreeSHAP explainability (`explain_gbpp()`).
* Point process quadrature and data constructors for 2D planar point patterns
  (`prepare_gbpp()`) and 1D linear network point patterns
  (`prepare_gbpp_network()`, `prepare_quad_data()`, `new_gbpp_data()`,
  `validate_gbpp_data()`).
* Six point process objectives: `"poisson"`, `"weighted_poisson"`,
  `"logistic"`, `"weighted_logistic"`, `"iterative_poisson"`, and
  `"iterative_logistic"`, with helpers `attach_stat_weight()`,
  `est_interaction()` / `attach_interaction()`, and
  `est_net_interaction()` / `attach_net_interaction()`.
* Independent stochastic thinning spatial cross-validation (`cv_gbpp()`,
  `plot.gbpp_cv()`, `calc_poisson_loglik()`, `calc_poisson_deviance()`).
* Python environment setup and diagnostics (`install_gbpp_python()`,
  `configure_gbpp_python()`, `check_gbpp_python()`, `get_gbpp_python()`).
