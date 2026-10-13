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
#' @section Uninformative observations:
#' When the prediction error variance \eqn{F_{t,i}} of an element is zero
#' to within rounding error (for example a series with zero loadings and
#' zero noise variance), that element carries no information about the
#' state and its update is skipped: `K[, i, t]` is zero and the smoother
#' passes over it.
#'
#' @param obs Observations \eqn{y}: an \eqn{n \times p}{n x p} numeric
#'   matrix, one row per time step.
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

  result <- list()

  n <- nrow(obs)
  p <- ncol(obs)
  m <- nrow(vartrans)
  q <- ncol(vartrans)

  F <- as.vector(t(matrix(x[[2]], p * n, m)))

  result$v <- t(x[[1]])
  result$F <- array(x[[2]], c(p, n))
  result$att <- t(x[[3]])
  result$Ptt <-  array(x[[4]], c(m, m, n))
  result$a <- t(x[[5]])
  result$P <-  array(x[[6]], c(m, m, n + 1))
  result$K <-  array(x[[7]] * F, c(m, p, n))
  result$r <- t(x[[8]])
  result$N <-  array(x[[9]], c(m, m, n + 1))
  result$epshat <- t(x[[10]])
  result$V_eps <- array(x[[11]], c(p, n))
  result$etahat <- t(x[[12]])
  result$V_eta <- array(x[[13]], c(q, q, n))
  result$alphahat <- t(x[[14]])
  result$V <-  array(x[[15]], c(m, m, n))

  return(result)
}

# Convert to double storage, keeping dimensions; the C code reads doubles
as_double <- function(x) {
  if (!is.numeric(x)) {
    stop(deparse(substitute(x)), " must be numeric", call. = FALSE)
  }
  storage.mode(x) <- "double"
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
#' @param n Number of draws.
#' @param mu Mean vector of length \eqn{d}.
#' @param sigma \eqn{d \times d}{d x d} positive definite covariance
#'   matrix.
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
  z <- .Call("mvrnorm", n, mu, sigma)

  return(t(z))
}
