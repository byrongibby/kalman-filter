#' kfilter: Kalman filter and smoother
#'
#' Kalman filtering and smoothing for linear Gaussian state space models,
#' implemented in C-style C++ on top of BLAS and LAPACK.
#'
#' * [kfas()] runs the filter, state smoother and disturbance smoother,
#'   with time-varying system matrices and non-diagonal observation noise.
#' * [kfas_loglik()] computes only the log-likelihood, for optimisation.
#' * [mvrnorm()] draws from a multivariate normal distribution.
#'
#' @keywords internal
#' @useDynLib kfilter
"_PACKAGE"
