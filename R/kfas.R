#' Kalman filter and smoother
#'
#' Runs the Kalman filter, state smoother and disturbance smoother for a
#' linear Gaussian state space model, using the univariate treatment of
#' multivariate series (Durbin and Koopman, 2012, section 6.4): each
#' element of \eqn{y_t} is processed in turn, so no matrix inversion is
#' needed.
#'
#' @section Model:
#' \deqn{y_t = d_t + Z_t \alpha_t + \varepsilon_t, \qquad
#'   \varepsilon_t \sim N(0, H_t)}{y[t] = d[t] + Z[t] alpha[t] + eps[t],
#'   eps[t] ~ N(0, H[t])}
#' \deqn{\alpha_{t+1} = c_t + T_t \alpha_t + R_t \eta_t, \qquad
#'   \eta_t \sim N(0, Q_t)}{alpha[t+1] = c[t] + T[t] alpha[t] + R[t] eta[t],
#'   eta[t] ~ N(0, Q[t])}
#' \deqn{\alpha_1 \sim N(a_1, P_{*} + \kappa P_\infty), \quad
#'   \kappa \to \infty}{alpha[1] ~ N(a1, P1 + kappa P1inf), kappa -> infinity}
#'
#' for \eqn{t = 1, \dots, n}, with \eqn{p} observed series, \eqn{m}
#' states and \eqn{q} state disturbances. \eqn{c_t}, \eqn{T_t},
#' \eqn{R_t} and \eqn{Q_t} take the state from time \eqn{t} to
#' \eqn{t + 1}, as in Durbin and Koopman and the KFAS package. The
#' intercepts \eqn{d_t} and \eqn{c_t} are optional (zero by default).
#'
#' @section Time-varying system matrices:
#' Each of `obsmat`, `obsvar`, `statemat`, `vartrans` and `statevar` may
#' be a single matrix (time-invariant) or an array with one slice per time
#' step, e.g. `obsmat` of dimension `c(p, m, n)` with `obsmat[, , t]` the
#' value of \eqn{Z_t}. Time-invariant and time-varying matrices can be
#' mixed freely. An input whose length matches neither form is an error.
#'
#' The intercepts `obsint` (\eqn{d_t}) and `stateint` (\eqn{c_t}) are
#' likewise a single vector (time-invariant) or a matrix with one row per
#' time step, e.g. `obsint` of dimension `c(n, p)` with `obsint[t, ]` the
#' value of \eqn{d_t}, laid out like `obs`.
#'
#' @section Intercepts:
#' The intercepts change only the means: the filter uses
#' \eqn{v_t = y_t - d_t - Z_t a_t} and
#' \eqn{a_{t+1} = c_t + T_t a_{t|t}}, and the variances, `K`, `r` and `N`
#' do not depend on them. Regression effects on the observations can be
#' given as `obsint` \eqn{= X_t \beta}{= X[t] beta} for known
#' \eqn{\beta}{beta}, without adding them to the state.
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
#' @section Diffuse initialisation:
#' Elements of the initial state with unknown starting values (for example
#' the level of a non-stationary series) are given a diffuse prior through
#' `initpinf`: \eqn{P_1 = P_{*} + \kappa P_\infty} with
#' \eqn{\kappa \to \infty}. The filter and smoother use the exact diffuse
#' initialisation of Durbin and Koopman (2012, chapter 5) in its univariate
#' form, so no large-variance approximation is needed. Typically
#' `initpinf` has ones on the diagonal for the diffuse elements and zeros
#' elsewhere, and `initp` has zero rows and columns for them.
#'
#' The diffuse phase lasts until \eqn{P_\infty} becomes zero, which ends
#' at series `j` of time step `d` (both returned). During it, `F` holds
#' \eqn{F_{*}} and `Finf` holds \eqn{F_\infty}; observations with
#' \eqn{F_\infty > 0} contribute \eqn{-\frac{1}{2} \log F_\infty}{-log(Finf) / 2}
#' to the diffuse log-likelihood, without the \eqn{\log 2\pi}{log(2 pi)}
#' constant, as in KFAS.
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
#' @param initp Variance of the initial state, \eqn{P_1} (or its
#'   non-diffuse part \eqn{P_{*}}): \eqn{m \times m}{m x m}.
#' @param initpinf Diffuse part of the initial state variance,
#'   \eqn{P_\infty}: \eqn{m \times m}{m x m}, or `NULL` (the default)
#'   for no diffuse elements. See the Diffuse initialisation section.
#' @param obsint Observation intercept \eqn{d}: a length-\eqn{p} vector,
#'   or an \eqn{n \times p}{n x p} matrix if time-varying. `NULL` (the
#'   default) for none.
#' @param stateint State intercept \eqn{c}: a length-\eqn{m} vector, or
#'   an \eqn{n \times m}{n x m} matrix if time-varying, with row \eqn{t}
#'   entering \eqn{\alpha_{t+1}}{alpha[t+1]}. `NULL` (the default) for
#'   none.
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
#'   \item{`Pinf`}{\eqn{m \times m \times d}{m x m x d} diffuse parts
#'     \eqn{P_{\infty,t}} of the predicted state variances over the
#'     diffuse phase.}
#'   \item{`Finf`}{\eqn{p \times d}{p x d} diffuse parts
#'     \eqn{F_{\infty,t,i}} of the prediction error variances.}
#'   \item{`d`, `j`}{The diffuse phase ends at series `j` of time step `d`
#'     (both 0 without diffuse elements).}
#'   \item{`logLik`}{The Gaussian log-likelihood
#'     \eqn{\log p(y_1, \dots, y_n)}, including the
#'     \eqn{-\frac{1}{2} \log 2\pi}{-log(2 pi) / 2} constant per
#'     observation. Missing and uninformative observations (see above) do
#'     not contribute. For non-diagonal \eqn{H} it includes the Jacobian of the
#'     Cholesky transform, so it is the likelihood of the original
#'     \eqn{y}. With a diffuse initial state it is the diffuse
#'     log-likelihood (see the Diffuse initialisation section). Computed by the filter, whatever `smoother` and
#'     `disturbances` are. Use [kfas_loglik()] when only the
#'     log-likelihood is needed.}
#' }
#'
#' @references Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis
#'   by State Space Methods*, 2nd edition. Oxford University Press.
#'
#' @examples
#' # Local level model for the Nile data, with the maximum likelihood
#' # variances from Durbin and Koopman (2012, section 2.10) and a diffuse
#' # initial level
#' y <- matrix(as.numeric(Nile))
#' fit <- kfas(y, obsmat = matrix(1), obsvar = matrix(15099),
#'             statemat = matrix(1), vartrans = matrix(1),
#'             statevar = matrix(1469.1), inita = 0, initp = matrix(0),
#'             initpinf = matrix(1))
#' plot(as.numeric(Nile), type = "p", ylab = "Flow")
#' lines(fit$alphahat[, 1])
#'
#' # The same model with the level variance doubled from 1899 onwards,
#' # using a time-varying Q
#' n <- nrow(y)
#' Q <- array(1469.1, c(1, 1, n))
#' Q[1, 1, 29:n] <- 2 * 1469.1
#' fit_tv <- kfas(y, matrix(1), matrix(15099), matrix(1), matrix(1), Q,
#'                0, matrix(0), matrix(1))
#'
#' # A known level shift of -250 from 1899 onwards, given as an
#' # observation intercept (an n x p matrix when time-varying)
#' shift <- ifelse(seq_len(n) >= 29, -250, 0)
#' fit_shift <- kfas(y, matrix(1), matrix(15099), matrix(1), matrix(1),
#'                   matrix(1469.1), 0, matrix(0), matrix(1),
#'                   obsint = matrix(shift))
#'
#' # Filtering only
#' filtered <- kfas(y, matrix(1), matrix(15099), matrix(1), matrix(1),
#'                  matrix(1469.1), 0, matrix(0), matrix(1),
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
  initpinf = NULL,
  obsint = NULL,
  stateint = NULL,
  smoother = TRUE,
  disturbances = TRUE
) {
  if (!is.matrix(obs)) stop("obs must be an n x p matrix", call. = FALSE)
  if (is.null(initpinf)) initpinf <- diffuse_none(vartrans)

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
    as_double(initpinf),
    intercept(obsint),
    intercept(stateint),
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
    logLik = x[[16]],
    Pinf = x[[17]],
    Finf = x[[18]],
    d = x[[19]],
    j = x[[20]]
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
#' The model, arguments, time-varying system matrices, intercepts,
#' non-diagonal \eqn{H}, missing observations and diffuse initialisation
#' are as described in [kfas()].
#'
#' @inheritParams kfas
#'
#' @return The log-likelihood \eqn{\log p(y_1, \dots, y_n)}, a single
#'   number. See `logLik` in [kfas()] for details.
#'
#' @examples
#' # Maximum likelihood estimation of the two variances of the local level
#' # model for the Nile data, with a diffuse initial level
#' y <- matrix(as.numeric(Nile))
#' negloglik <- function(log_var) {
#'   -kfas_loglik(y, matrix(1), matrix(exp(log_var[1])), matrix(1),
#'                matrix(1), matrix(exp(log_var[2])), 0, matrix(0),
#'                matrix(1))
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
  initp,
  initpinf = NULL,
  obsint = NULL,
  stateint = NULL
) {
  if (!is.matrix(obs)) stop("obs must be an n x p matrix", call. = FALSE)
  if (is.null(initpinf)) initpinf <- diffuse_none(vartrans)

  .Call(
    "kfas0_loglik",
    t(as_double(obs)),
    as_double(obsmat),
    as_double(obsvar),
    as_double(statemat),
    as_double(vartrans),
    as_double(statevar),
    as_double(inita),
    as_double(initp),
    as_double(initpinf),
    intercept(obsint),
    intercept(stateint)
  )
}

# No diffuse initial elements: an m x m zero matrix, m from R (vartrans)
diffuse_none <- function(vartrans) {
  m <- if (is.null(dim(vartrans))) 1 else dim(vartrans)[1]
  matrix(0, m, m)
}

# Intercepts for the C code: NULL for none, otherwise doubles with each time
# step's values contiguous (a time-varying intercept is given time by row,
# like obs)
intercept <- function(x) {
  if (is.null(x)) return(NULL)
  if (!is.numeric(x)) {
    stop(deparse(substitute(x)), " must be numeric", call. = FALSE)
  }
  if (is.matrix(x)) x <- t(x)
  storage.mode(x) <- "double"
  x
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

#' Precision-based sampler for the states
#'
#' Draws the states \eqn{\alpha_1, \dots, \alpha_n} jointly from their
#' posterior distribution given the observations, using the precision-based
#' sampler of Chan and Jeliazkov (2009). Instead of running the Kalman
#' filter and a backward recursion, it writes the posterior of the stacked
#' state vector \eqn{\alpha = (\alpha_1', \dots, \alpha_n')'}{alpha =
#' (alpha[1]', ..., alpha[n]')'} as \eqn{N(\mu, K^{-1})}{N(mu, K^-1)}. The
#' precision \eqn{K} is block tridiagonal, so it is factored as a band
#' matrix in \eqn{O(n m^3)} operations, and each draw is then
#' \eqn{\mu + L'^{-1} z}{mu + L'^-1 z}, with \eqn{K = L L'} and \eqn{z}
#' standard normal, at the cost of a banded triangular solve.
#'
#' The model, arguments, time-varying system matrices, intercepts and
#' missing observations are as described in [kfas()]. Non-diagonal
#' \eqn{H} is allowed.
#'
#' @section Requirements:
#' The posterior precision involves the inverses of the variances in the
#' model, so the sampler needs
#' * \eqn{R_t Q_t R_t'} positive definite for \eqn{t = 1, \dots, n - 1}:
#'   every state has its own disturbance. Models with fewer disturbances
#'   than states (\eqn{q < m}), such as those with a deterministic trend
#'   slope, are not supported;
#' * \eqn{H_t} positive definite (over the observed elements);
#' * `initp` positive definite, apart from diffuse elements given by
#'   `initpinf`. Diffuse elements have a flat prior, the limit of the
#'   exact diffuse initialisation, so the draws come from the same
#'   posterior that [kfas()] smooths with a diffuse initial state.
#'
#' For models outside these conditions use the Kalman filter based
#' smoother in [kfas()] for the posterior mean and variances.
#'
#' @section Linear constraints:
#' `constraints` and `rhs` restrict the draws to the hyperplane
#' \eqn{A \alpha = r}{A alpha = r}, for example states that sum to zero
#' or quarterly states that add up to known annual totals. The draws then
#' come exactly from the posterior conditional on the constraints, using
#' algorithm 2 of Cong, Chen and Zhou (2017): each unconstrained draw
#' \eqn{x} is corrected to
#' \eqn{x + K^{-1} A' (A K^{-1} A')^{-1} (r - A x)}{x + K^-1 A' (A K^-1
#' A')^-1 (r - A x)}. The \eqn{k} solves with the banded \eqn{K} are done
#' once, so each draw costs an extra \eqn{O(n m k)} operations. `mean` is
#' then the conditional posterior mean.
#'
#' Row \eqn{j} of `constraints` holds the coefficients of constraint
#' \eqn{j} on `as.vector(alpha)`, where `alpha` is the \eqn{n \times m}{n
#' x m} matrix of states laid out like `alphahat`, so that
#' `constraints %*% as.vector(sim$draws[, , s])` equals `rhs`. The
#' constraints must be linearly independent.
#'
#' Draws use a 64-bit Mersenne Twister seeded from R's random number
#' generator, as in [mvrnorm()], so [set.seed()] makes them reproducible.
#'
#' @inheritParams kfas
#' @param nsim Number of draws, a positive integer.
#' @param constraints Linear constraints on the states: a
#'   \eqn{k \times n m}{k x nm} matrix (or a \eqn{k \times n \times m}{k x
#'   n x m} array, or a vector of length \eqn{n m} for one constraint).
#'   `NULL` (the default) for none. See the Linear constraints section.
#' @param rhs Right-hand side \eqn{r} of the constraints, a
#'   length-\eqn{k} vector. Zero if `NULL`.
#'
#' @return A list with elements
#' \describe{
#'   \item{`draws`}{\eqn{n \times m \times}{n x m x} `nsim` array of draws:
#'     `draws[, , s]` is one draw of the states, laid out like `alphahat`
#'     in [kfas()].}
#'   \item{`mean`}{\eqn{n \times m}{n x m} posterior mean
#'     \eqn{E(\alpha_t | y)}, equal to `alphahat` from [kfas()] (with
#'     constraints, the mean conditional on them).}
#' }
#'
#' @references Chan, J. C. C. and Jeliazkov, I. (2009). Efficient
#'   simulation and integrated likelihood estimation in state space models.
#'   *International Journal of Mathematical Modelling and Numerical
#'   Optimisation*, 1(1/2), 101--120.
#'
#'   Cong, Y., Chen, B. and Zhou, M. (2017). Fast simulation of
#'   hyperplane-truncated multivariate normal distributions. *Bayesian
#'   Analysis*, 12(4), 1017--1037.
#'
#' @examples
#' # Draws of the level of the Nile, with a diffuse initial level
#' y <- matrix(as.numeric(Nile))
#' set.seed(1)
#' sim <- precision_sampler(y, matrix(1), matrix(15099), matrix(1),
#'                          matrix(1), matrix(1469.1), 0, matrix(0),
#'                          matrix(1), nsim = 100)
#' plot(as.numeric(Nile), ylab = "Flow")
#' matlines(sim$draws[, 1, 1:10], col = "grey", lty = 1)
#' lines(sim$mean[, 1])
#'
#' # The same, constrained so that the level averages 900 over each decade
#' n <- nrow(y)
#' decades <- split(seq_len(n), (seq_len(n) - 1) %/% 10)
#' A <- t(sapply(decades, function(i) as.numeric(seq_len(n) %in% i) / 10))
#' sim_c <- precision_sampler(y, matrix(1), matrix(15099), matrix(1),
#'                            matrix(1), matrix(1469.1), 0, matrix(0),
#'                            matrix(1), nsim = 100, constraints = A,
#'                            rhs = rep(900, nrow(A)))
#' range(A %*% sim_c$draws[, 1, ])
#'
#' @export
precision_sampler <- function(
  obs,
  obsmat,
  obsvar,
  statemat,
  vartrans,
  statevar,
  inita,
  initp,
  initpinf = NULL,
  obsint = NULL,
  stateint = NULL,
  nsim = 1,
  constraints = NULL,
  rhs = NULL
) {
  if (!is.matrix(obs)) stop("obs must be an n x p matrix", call. = FALSE)
  if (is.null(initpinf)) initpinf <- diffuse_none(vartrans)

  # Constraints on as.vector(alpha) (time fastest) are reordered to the
  # stacked states of the C code (state fastest)
  if (is.null(constraints) && !is.null(rhs)) {
    stop("rhs is given without constraints", call. = FALSE)
  }
  if (!is.null(constraints)) {
    constraints <- as_double(constraints)
    n <- nrow(obs)
    m <- if (is.null(dim(vartrans))) 1 else dim(vartrans)[1]
    if (length(constraints) %% (n * m) != 0) {
      stop("constraints must have n * m = ", n * m, " columns",
           call. = FALSE)
    }
    k <- length(constraints) / (n * m)
    constraints <- matrix(aperm(array(constraints, c(k, n, m)), c(1, 3, 2)),
                          k)
    rhs <- if (is.null(rhs)) rep(0, k) else as_double(rhs)
    if (length(rhs) != k) {
      stop("rhs must have one element per constraint (", k, ")",
           call. = FALSE)
    }
  }

  x <- .Call(
    "precision_sampler",
    t(as_double(obs)),
    as_double(obsmat),
    as_double(obsvar),
    as_double(statemat),
    as_double(vartrans),
    as_double(statevar),
    as_double(inita),
    as_double(initp),
    as_double(initpinf),
    intercept(obsint),
    intercept(stateint),
    nsim,
    constraints,
    rhs
  )

  list(draws = aperm(x[[1]], c(2, 1, 3)), mean = t(x[[2]]))
}

#' Simulation smoother for the states
#'
#' Draws the states \eqn{\alpha_1, \dots, \alpha_n} jointly from their
#' posterior distribution given the observations, using the simulation
#' smoother of Durbin and Koopman (2002). Each draw is
#' \eqn{\alpha^+ + \hat\alpha - \hat\alpha^+}{alpha+ + alphahat -
#' alphahat+}, where \eqn{(\alpha^+, y^+)}{(alpha+, y+)} is simulated from
#' the model and \eqn{\hat\alpha^+}{alphahat+} is the smoothed state for
#' \eqn{y^+}{y+}.
#'
#' The model, arguments, time-varying system matrices, intercepts,
#' non-diagonal \eqn{H}, missing observations and diffuse initialisation
#' are as described in [kfas()], and every model [kfas()] accepts is
#' supported, including singular \eqn{R_t Q_t R_t'}{R[t] Q[t] R[t]'},
#' \eqn{H} or `initp`.
#'
#' The filter variances do not depend on the data, so they are computed
#' once; each draw then costs only the mean recursions of the filter and
#' smoother, \eqn{O(n (p m + m^2))}{O(n (p m + m^2))} operations. When the
#' model meets its requirements, [precision_sampler()] draws from the same
#' distribution and is usually faster for many draws of small models.
#'
#' Diffuse initial elements of \eqn{\alpha^+}{alpha+} are drawn at `inita`:
#' the smoothed state is invariant to shifts of the initial state in the
#' diffuse directions, so their value does not matter. Draws use a 64-bit
#' Mersenne Twister seeded from R's random number generator, as in
#' [mvrnorm()], so [set.seed()] makes them reproducible.
#'
#' @inheritParams kfas
#' @param nsim Number of draws, a positive integer.
#'
#' @return A list with elements
#' \describe{
#'   \item{`draws`}{\eqn{n \times m \times}{n x m x} `nsim` array of draws:
#'     `draws[, , s]` is one draw of the states, laid out like `alphahat`
#'     in [kfas()].}
#'   \item{`mean`}{\eqn{n \times m}{n x m} posterior mean
#'     \eqn{E(\alpha_t | y)}, equal to `alphahat` from [kfas()].}
#' }
#'
#' @references Durbin, J. and Koopman, S. J. (2002). A simple and
#'   efficient simulation smoother for state space time series analysis.
#'   *Biometrika*, 89(3), 603--615.
#'
#' @examples
#' # Draws of the level of the Nile, with a diffuse initial level
#' y <- matrix(as.numeric(Nile))
#' set.seed(1)
#' sim <- simulation_smoother(y, matrix(1), matrix(15099), matrix(1),
#'                            matrix(1), matrix(1469.1), 0, matrix(0),
#'                            matrix(1), nsim = 100)
#' plot(as.numeric(Nile), ylab = "Flow")
#' matlines(sim$draws[, 1, 1:10], col = "grey", lty = 1)
#' lines(sim$mean[, 1])
#'
#' @export
simulation_smoother <- function(
  obs,
  obsmat,
  obsvar,
  statemat,
  vartrans,
  statevar,
  inita,
  initp,
  initpinf = NULL,
  obsint = NULL,
  stateint = NULL,
  nsim = 1
) {
  if (!is.matrix(obs)) stop("obs must be an n x p matrix", call. = FALSE)
  if (is.null(initpinf)) initpinf <- diffuse_none(vartrans)

  x <- .Call(
    "kfas0_simulate",
    t(as_double(obs)),
    as_double(obsmat),
    as_double(obsvar),
    as_double(statemat),
    as_double(vartrans),
    as_double(statevar),
    as_double(inita),
    as_double(initp),
    as_double(initpinf),
    intercept(obsint),
    intercept(stateint),
    nsim
  )

  list(draws = aperm(x[[1]], c(2, 1, 3)), mean = t(x[[2]]))
}
