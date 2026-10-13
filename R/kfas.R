#' Kalman filter and smoother
#'
#' Runs the Kalman filter, state smoother and disturbance smoother for a
#' linear Gaussian state space model, using the univariate treatment of
#' multivariate series (Durbin and Koopman, 2012, section 6.4): each
#' element of \eqn{y_t} is processed in turn, so no matrix inversion is
#' needed.
#'
#' @section Model:
#' \deqn{y_t = Z_t \alpha_t + \varepsilon_t, \qquad
#'   \varepsilon_t \sim N(0, H_t)}{y[t] = Z[t] alpha[t] + eps[t],
#'   eps[t] ~ N(0, H[t])}
#' \deqn{\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, \qquad
#'   \eta_t \sim N(0, Q_t)}{alpha[t+1] = T[t] alpha[t] + R[t] eta[t],
#'   eta[t] ~ N(0, Q[t])}
#' \deqn{\alpha_1 \sim N(a_1, P_1)}{alpha[1] ~ N(a1, P1)}
#'
#' for \eqn{t = 1, \dots, n}, with \eqn{p} observed series, \eqn{m}
#' states and \eqn{q} state disturbances. \eqn{T_t}, \eqn{R_t} and
#' \eqn{Q_t} take the state from time \eqn{t} to \eqn{t + 1}, as in
#' Durbin and Koopman and the KFAS package.
#'
#' @section Time-varying system matrices:
#' Each of `obsmat`, `obsvar`, `statemat`, `vartrans` and `statevar` may
#' be a single matrix (time-invariant) or an array with one slice per time
#' step, e.g. `obsmat` of dimension `c(p, m, n)` with `obsmat[, , t]` the
#' value of \eqn{Z_t}. Time-invariant and time-varying matrices can be
#' mixed freely. An input whose length matches neither form is an error.
#'
#' @section Non-diagonal H:
#' The univariate treatment requires diagonal \eqn{H_t}. When any slice of
#' `obsvar` has a non-zero off-diagonal element, each \eqn{H_t} is
#' factored as \eqn{L_t L_t'} (Cholesky) and the model is filtered with
#' \eqn{y_t^* = L_t^{-1} y_t}, \eqn{Z_t^* = L_t^{-1} Z_t} and
#' \eqn{H_t^* = I}. This affects the outputs as follows:
#' * the states and state disturbances (`att`, `Ptt`, `a`, `P`, `r`, `N`,
#'   `alphahat`, `V`, `etahat`, `V_eta`) are unaffected;
#' * `v`, `F` and `K` refer to the transformed series \eqn{y_t^*};
#' * `epshat` and `V_eps` are returned in the original scale of
#'   \eqn{y_t}.
#'
#' KFAS uses an LDL decomposition instead, so for non-diagonal \eqn{H} its
#' `v`, `F`, `epshat` and `V_eps` are on a different scale.
#'
#' @section Missing observations:
#' `NA` (or `NaN`) elements of `obs` are treated as missing: the filter
#' skips them and every output conditions on the observed elements only.
#' Any pattern is allowed, including time steps with nothing observed.
#' At a missing \eqn{y_{t,i}}, `v`, `F` and `K` are `NA`; `epshat` and
#' `V_eps` give the conditional mean and variance of
#' \eqn{\varepsilon_{t,i}} given the observed data (0 and
#' \eqn{H_{t,ii}} when \eqn{H_t} is diagonal). With non-diagonal
#' \eqn{H_t}, the Cholesky transform at each time step uses only the
#' observed elements.
#'
#' @section Uninformative observations:
#' When the prediction error variance \eqn{F_{t,i}} of an element is zero
#' to within rounding error (for example a series with zero loadings and
#' zero noise variance), that element carries no information about the
#' state and its update is skipped: `K[, i, t]` is zero and the smoother
#' passes over it.
#'
#' @param obs Observations \eqn{y}: an \eqn{n \times p}{n x p} numeric
#'   matrix, one row per time step. `NA` marks missing observations.
#' @param obsmat Observation matrix \eqn{Z}: \eqn{p \times m}{p x m}, or
#'   \eqn{p \times m \times n}{p x m x n} if time-varying.
#' @param obsvar Observation noise variance \eqn{H}:
#'   \eqn{p \times p}{p x p}, or \eqn{p \times p \times n}{p x p x n}.
#'   Must be positive definite if not diagonal.
#' @param statemat Transition matrix \eqn{T}: \eqn{m \times m}{m x m}, or
#'   \eqn{m \times m \times n}{m x m x n}.
#' @param vartrans Disturbance selection matrix \eqn{R}:
#'   \eqn{m \times q}{m x q}, or \eqn{m \times q \times n}{m x q x n}.
#'   Its dimensions define \eqn{m} and \eqn{q}.
#' @param statevar State disturbance variance \eqn{Q}:
#'   \eqn{q \times q}{q x q}, or \eqn{q \times q \times n}{q x q x n}.
#' @param inita Mean of the initial state \eqn{a_1}: a length-\eqn{m}
#'   vector.
#' @param initp Variance of the initial state \eqn{P_1}:
#'   \eqn{m \times m}{m x m}. A diffuse initial state can be approximated
#'   with a large multiple of the identity.
#' @param smoother If `TRUE` (the default), compute the smoothed state
#'   `alphahat` and its variance `V`. Otherwise they are `NA`.
#' @param disturbances If `TRUE` (the default), compute the smoothed
#'   disturbances `epshat`, `etahat` and their variances. Otherwise they
#'   are `NA`. `r` and `N` are computed when either `smoother` or
#'   `disturbances` is `TRUE`.
#'
#' @return A list with elements, where \eqn{i} indexes the observed series
#'   within time step \eqn{t}:
#' \describe{
#'   \item{`v`}{\eqn{n \times p}{n x p} prediction errors \eqn{v_{t,i}}.}
#'   \item{`F`}{\eqn{p \times n}{p x n} prediction error variances
#'     \eqn{F_{t,i}}.}
#'   \item{`att`}{\eqn{n \times m}{n x m} filtered states
#'     \eqn{a_{t|t} = E(\alpha_t | y_1, \dots, y_t)}.}
#'   \item{`Ptt`}{\eqn{m \times m \times n}{m x m x n} filtered state
#'     variances \eqn{P_{t|t}}.}
#'   \item{`a`}{\eqn{(n + 1) \times m}{(n + 1) x m} predicted states
#'     \eqn{a_t = E(\alpha_t | y_1, \dots, y_{t-1})}, including the
#'     one-step-ahead forecast \eqn{a_{n+1}}.}
#'   \item{`P`}{\eqn{m \times m \times (n + 1)}{m x m x (n + 1)} predicted
#'     state variances \eqn{P_t}.}
#'   \item{`K`}{\eqn{m \times p \times n}{m x p x n} covariances
#'     \eqn{P_{t,i} Z_{t,i}'} between the state and each \eqn{y_{t,i}}
#'     given the observations processed before it.}
#'   \item{`r`}{\eqn{(n + 1) \times m}{(n + 1) x m} smoothing cumulants:
#'     row \eqn{t} is the weighted sum of innovations from time \eqn{t}
#'     onwards (\eqn{r_{t-1}} in Durbin and Koopman's notation); row
#'     \eqn{n + 1} is zero.}
#'   \item{`N`}{\eqn{m \times m \times (n + 1)}{m x m x (n + 1)} variances
#'     of `r`.}
#'   \item{`epshat`}{\eqn{n \times p}{n x p} smoothed observation
#'     disturbances \eqn{E(\varepsilon_t | y)}.}
#'   \item{`V_eps`}{\eqn{p \times n}{p x n} diagonals of
#'     \eqn{Var(\varepsilon_t | y)}.}
#'   \item{`etahat`}{\eqn{n \times q}{n x q} smoothed state disturbances
#'     \eqn{E(\eta_t | y)}.}
#'   \item{`V_eta`}{\eqn{q \times q \times n}{q x q x n}
#'     \eqn{Var(\eta_t | y)}.}
#'   \item{`alphahat`}{\eqn{n \times m}{n x m} smoothed states
#'     \eqn{E(\alpha_t | y)}.}
#'   \item{`V`}{\eqn{m \times m \times n}{m x m x n} smoothed state
#'     variances \eqn{Var(\alpha_t | y)}.}
#'   \item{`logLik`}{The Gaussian log-likelihood
#'     \eqn{\log p(y_1, \dots, y_n)}, including the
#'     \eqn{-\frac{1}{2} \log 2\pi}{-log(2 pi) / 2} constant per
#'     observation. Missing and uninformative observations (see above) do
#'     not contribute. For non-diagonal \eqn{H} it includes the Jacobian of the
#'     Cholesky transform, so it is the likelihood of the original
#'     \eqn{y}. Computed by the filter, whatever `smoother` and
#'     `disturbances` are. Use [kfas_loglik()] when only the
#'     log-likelihood is needed.}
#' }
#'
#' @references Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis
#'   by State Space Methods*, 2nd edition. Oxford University Press.
#'
#' @examples
#' # Local level model for the Nile data, with the maximum likelihood
#' # variances from Durbin and Koopman (2012, section 2.10)
#' y <- matrix(as.numeric(Nile))
#' fit <- kfas(y, obsmat = matrix(1), obsvar = matrix(15099),
#'             statemat = matrix(1), vartrans = matrix(1),
#'             statevar = matrix(1469.1), inita = 0, initp = matrix(1e7))
#' plot(as.numeric(Nile), type = "p", ylab = "Flow")
#' lines(fit$alphahat[, 1])
#'
#' # The same model with the level variance doubled from 1899 onwards,
#' # using a time-varying Q
#' n <- nrow(y)
#' Q <- array(1469.1, c(1, 1, n))
#' Q[1, 1, 29:n] <- 2 * 1469.1
#' fit_tv <- kfas(y, matrix(1), matrix(15099), matrix(1), matrix(1), Q,
#'                0, matrix(1e7))
#'
#' # Filtering only
#' filtered <- kfas(y, matrix(1), matrix(15099), matrix(1), matrix(1),
#'                  matrix(1469.1), 0, matrix(1e7),
#'                  smoother = FALSE, disturbances = FALSE)
#'
#' @export
kfas <- function(
  obs,
  obsmat,
  obsvar,
  statemat,
  vartrans,
  statevar,
  inita,
  initp,
  smoother = TRUE,
  disturbances = TRUE
) {
  if (!is.matrix(obs)) stop("obs must be an n x p matrix", call. = FALSE)

  x <- .Call(
    "kfas0",
    t(as_double(obs)),
    as_double(obsmat),
    as_double(obsvar),
    as_double(statemat),
    as_double(vartrans),
    as_double(statevar),
    as_double(inita),
    as_double(initp),
    as.logical(smoother),
    as.logical(disturbances)
  )

  # The C code returns arrays already shaped; series-by-time matrices are
  # transposed to time-by-series
  list(
    v = t(x[[1]]),
    F = x[[2]],
    att = t(x[[3]]),
    Ptt = x[[4]],
    a = t(x[[5]]),
    P = x[[6]],
    K = x[[7]],
    r = t(x[[8]]),
    N = x[[9]],
    epshat = t(x[[10]]),
    V_eps = x[[11]],
    etahat = t(x[[12]]),
    V_eta = x[[13]],
    alphahat = t(x[[14]]),
    V = x[[15]],
    logLik = x[[16]]
  )
}

#' Log-likelihood of a state space model
#'
#' Computes the Gaussian log-likelihood of a linear Gaussian state space
#' model with the Kalman filter, without storing any per-step output. It
#' returns the same value as `kfas(...)$logLik` using O(\eqn{m^2}) rather
#' than O(\eqn{n m^2}) memory, and is faster, which matters when it is
#' evaluated many times, for example inside an optimiser.
#'
#' The model, arguments, time-varying system matrices, non-diagonal
#' \eqn{H} and missing observations are as described in [kfas()].
#'
#' @inheritParams kfas
#'
#' @return The log-likelihood \eqn{\log p(y_1, \dots, y_n)}, a single
#'   number. See `logLik` in [kfas()] for details.
#'
#' @examples
#' # Maximum likelihood estimation of the two variances of the local level
#' # model for the Nile data. (The large initp approximates a diffuse
#' # initial state, so the first observation's contribution depends on it.)
#' y <- matrix(as.numeric(Nile))
#' negloglik <- function(log_var) {
#'   -kfas_loglik(y, matrix(1), matrix(exp(log_var[1])), matrix(1),
#'                matrix(1), matrix(exp(log_var[2])), 0, matrix(1e7))
#' }
#' fit <- optim(c(log(var(y)), log(var(y))), negloglik)
#' exp(fit$par)
#'
#' @export
kfas_loglik <- function(
  obs,
  obsmat,
  obsvar,
  statemat,
  vartrans,
  statevar,
  inita,
  initp
) {
  if (!is.matrix(obs)) stop("obs must be an n x p matrix", call. = FALSE)

  .Call(
    "kfas0_loglik",
    t(as_double(obs)),
    as_double(obsmat),
    as_double(obsvar),
    as_double(statemat),
    as_double(vartrans),
    as_double(statevar),
    as_double(inita),
    as_double(initp)
  )
}

# Convert to double storage, keeping dimensions; the C code reads doubles
as_double <- function(x) {
  if (!is.numeric(x)) {
    stop(deparse(substitute(x)), " must be numeric", call. = FALSE)
  }
  if (!is.double(x)) storage.mode(x) <- "double"
  x
}

#' Multivariate normal random draws
#'
#' Draws from a multivariate normal distribution using the Cholesky factor
#' of the covariance matrix. Draws come from a 64-bit Mersenne Twister
#' seeded from R's random number generator, so [set.seed()] makes them
#' reproducible (on a given platform: the C++ standard does not fix the
#' exact output of the normal distribution across standard libraries).
#'
#' @param n Number of draws, a positive integer.
#' @param mu Mean vector of length \eqn{d}.
#' @param sigma \eqn{d \times d}{d x d} symmetric positive definite
#'   covariance matrix. An error is raised if it is not.
#'
#' @return An \eqn{n \times d}{n x d} matrix, one draw per row.
#'
#' @examples
#' set.seed(1)
#' sigma <- matrix(c(2, 0.5, 0.5, 1), 2)
#' x <- mvrnorm(1000, c(0, 1), sigma)
#' colMeans(x)
#' cov(x)
#'
#' @export
mvrnorm <- function(n, mu, sigma) {
  sigma <- as_double(sigma)
  # Only the lower triangle is used, so an asymmetric sigma would be
  # silently misread
  if (is.matrix(sigma) && nrow(sigma) == ncol(sigma) &&
      !isSymmetric(unname(sigma))) {
    stop("sigma must be symmetric", call. = FALSE)
  }

  z <- .Call("mvrnorm", n, as_double(mu), sigma)

  return(t(z))
}
