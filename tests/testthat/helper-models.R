# Test models in the notation of Durbin & Koopman:
#   y_t       = Z alpha_t + eps_t,        eps_t ~ N(0, H)
#   alpha_t+1 = T alpha_t + R eta_t,      eta_t ~ N(0, Q)
#   alpha_1   ~ N(a1, P1)
# H must be diagonal for kfas() (univariate treatment).

local_level <- function() {
  list(Z = matrix(1), H = matrix(1), T = matrix(1), R = matrix(1),
       Q = matrix(1), a1 = 0, P1 = matrix(10))
}

local_trend <- function() {
  list(Z = matrix(c(1, 0), 1), H = matrix(1),
       T = matrix(c(1, 0, 1, 1), 2), R = diag(2), Q = diag(c(0.5, 0.1)),
       a1 = c(0, 0), P1 = diag(2) * 10)
}

# p = 2 series, m = 3 states, q = 2 disturbances, with a full Z so every
# state enters every observation equation
bivariate <- function() {
  list(Z = matrix(c(1, 0.5, 0.3, 1, -0.2, 0.4), 2), H = diag(c(1, 2)),
       T = matrix(c(0.9, 0, 0, 0.1, 0.7, 0, 0, 0.2, 0.5), 3),
       R = matrix(c(1, 0, 0.5, 0, 1, 0.5), 3), Q = diag(c(0.5, 0.3)),
       a1 = c(0, 0, 0), P1 = diag(3) * 10)
}

# Make the named system matrices time-varying: slice t is the base matrix
# scaled by a factor that varies with t (positive for H and Q, and keeping
# T stable)
time_varying <- function(mod, which, n = 40) {
  factors <- list(
    Z = function(t) 1 + 0.5 * sin(t),
    H = function(t) 1 + 0.8 * cos(t)^2,
    T = function(t) 0.8 + 0.2 * cos(t),
    R = function(t) 1 + 0.3 * sin(2 * t),
    Q = function(t) 0.5 + sin(t / 3)^2
  )
  for (name in which) {
    base <- mod[[name]]
    mod[[name]] <- array(
      vapply(seq_len(n), function(t) base * factors[[name]](t), base),
      c(dim(base), n))
  }
  mod
}

# Slice t of a system matrix, whether time-invariant or time-varying
at <- function(x, t) if (length(dim(x)) == 3) x[, , t] else x

simulate_y <- function(mod, n = 40, seed = 1) {
  set.seed(seed)
  p <- nrow(mod$Z)
  m <- ncol(mod$Z)
  q <- ncol(mod$R)
  y <- matrix(0, n, p)
  if (is.null(mod$P1inf)) {
    alpha <- mod$a1 + drop(t(chol(mod$P1)) %*% rnorm(m))
  } else {
    # Diffuse elements: draw the initial state with variance P1 + P1inf,
    # which may be singular
    e <- eigen(mod$P1 + mod$P1inf, symmetric = TRUE)
    alpha <- mod$a1 + drop(e$vectors %*% (sqrt(pmax(e$values, 0)) * rnorm(m)))
  }
  for (t in seq_len(n)) {
    y[t, ] <- at(mod$Z, t) %*% alpha +
      rnorm(p, sd = sqrt(diag(as.matrix(at(mod$H, t)))))
    alpha <- at(mod$T, t) %*% alpha +
      at(mod$R, t) %*% rnorm(q, sd = sqrt(diag(as.matrix(at(mod$Q, t)))))
  }
  y
}

run_kfas <- function(mod, y) {
  kfas(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1, mod$P1inf)
}

# The multivariate routine in src/kfas.cpp is not exported
run_kfas_multivariate <- function(mod, y, smoother = TRUE,
                                  disturbances = TRUE) {
  x <- .Call("kfas", t(y), mod$Z, mod$H, mod$T, mod$R, mod$Q,
             as.double(mod$a1), mod$P1, smoother, disturbances,
             PACKAGE = "kfilter")
  n <- nrow(y)
  p <- ncol(y)
  m <- dim(mod$R)[1]
  q <- dim(mod$R)[2]
  list(v = t(x[[1]]), F = array(x[[2]], c(p, p, n)), att = t(x[[3]]),
       Ptt = array(x[[4]], c(m, m, n)), a = t(x[[5]]),
       P = array(x[[6]], c(m, m, n + 1)), K = array(x[[7]], c(m, p, n)),
       r = t(x[[8]]), N = array(x[[9]], c(m, m, n + 1)),
       epshat = t(x[[10]]), V_eps = array(x[[11]], c(p, n)),
       etahat = t(x[[12]]), V_eta = array(x[[13]], c(q, q, n)),
       alphahat = t(x[[14]]), V = array(x[[15]], c(m, m, n)),
       logLik = x[[16]])
}

# Make the initial state diffuse: elements with diffuse = TRUE get
# P1inf = 1 and P1 = 0, the rest keep their P1
make_diffuse <- function(mod, diffuse = rep(TRUE, dim(mod$R)[1])) {
  mod$P1inf <- diag(as.numeric(diffuse), length(diffuse))
  mod$P1[diffuse, ] <- 0
  mod$P1[, diffuse] <- 0
  mod
}

# Replace H in a p = 2 model with a non-diagonal one
full_H <- function(mod) {
  mod$H <- matrix(c(1, 0.6, 0.6, 2), 2)
  mod
}

run_reference <- function(mod, y) {
  # SSModel only recognises SSMcustom by its bare name in the formula
  SSMcustom <- KFAS::SSMcustom
  m <- dim(mod$R)[1]
  P1inf <- if (is.null(mod$P1inf)) matrix(0, m, m) else mod$P1inf
  ssm <- KFAS::SSModel(
    y ~ -1 + SSMcustom(Z = mod$Z, T = mod$T, R = mod$R, Q = mod$Q,
                       a1 = mod$a1, P1 = mod$P1, P1inf = P1inf),
    H = mod$H)
  KFAS::KFS(ssm, filtering = "state", smoothing = c("state", "disturbance"))
}

# Compare values only: KFAS returns ts objects and differently shaped arrays
expect_same_values <- function(object, expected, tolerance = 1e-8) {
  expect_equal(as.vector(object), as.vector(expected), tolerance = tolerance)
}

# Exact smoothed quantities and log-likelihood from the joint normal
# distribution of the stacked model, conditioning on the observed (non-NA)
# elements of y only. alpha_t = A_t xi with xi = (alpha_1, eta_1, ...,
# eta_n-1), so y = B xi + eps with eps independent of xi.
exact_posterior <- function(mod, y) {
  n <- nrow(y)
  p <- ncol(y)
  m <- dim(mod$R)[1]
  q <- dim(mod$R)[2]
  k <- m + (n - 1) * q

  A <- vector("list", n)
  A[[1]] <- cbind(diag(m), matrix(0, m, k - m))
  E <- vector("list", n)  # eta_t = E_t xi
  B <- matrix(0, n * p, k)
  var_xi <- matrix(0, k, k)
  var_xi[1:m, 1:m] <- mod$P1
  var_eps <- matrix(0, n * p, n * p)
  for (t in seq_len(n)) {
    rows <- (t - 1) * p + seq_len(p)
    B[rows, ] <- at(mod$Z, t) %*% A[[t]]
    var_eps[rows, rows] <- at(mod$H, t)
    E[[t]] <- matrix(0, q, k)
    if (t < n) {
      cols <- m + (t - 1) * q + seq_len(q)
      var_xi[cols, cols] <- at(mod$Q, t)
      E[[t]][, cols] <- diag(q)
      A[[t + 1]] <- at(mod$T, t) %*% A[[t]] + at(mod$R, t) %*% E[[t]]
    }
  }

  y_vec <- as.vector(t(y))
  o <- !is.na(y_vec)
  mean_xi <- c(mod$a1, rep(0, k - m))
  S <- B[o, ] %*% var_xi %*% t(B[o, ]) + var_eps[o, o]
  e <- y_vec[o] - drop(B[o, ] %*% mean_xi)
  U <- chol(S)
  S_inv <- chol2inv(U)

  # Posterior of xi, and of eps (Cov(eps, y_o) = var_eps[, o])
  C_xi <- var_xi %*% t(B[o, ])
  post_mean <- mean_xi + drop(C_xi %*% S_inv %*% e)
  post_var <- var_xi - C_xi %*% S_inv %*% t(C_xi)
  C_eps <- var_eps[, o, drop = FALSE]
  eps_mean <- drop(C_eps %*% S_inv %*% e)
  eps_var <- diag(var_eps - C_eps %*% S_inv %*% t(C_eps))

  # eta_n is independent of y: its posterior is the prior
  Q_n <- at(mod$Q, n)
  list(
    alphahat = matrix(unlist(lapply(A, function(At) At %*% post_mean)),
                      n, m, byrow = TRUE),
    V = array(sapply(A, function(At) At %*% post_var %*% t(At)), c(m, m, n)),
    epshat = matrix(eps_mean, n, p, byrow = TRUE),
    V_eps = matrix(eps_var, p, n),
    etahat = matrix(c(unlist(lapply(E[-n], function(Et) Et %*% post_mean)),
                      rep(0, q)), n, q, byrow = TRUE),
    V_eta = array(c(sapply(E[-n], function(Et) Et %*% post_var %*% t(Et)),
                    Q_n), c(q, q, n)),
    logLik = -0.5 * (sum(o) * log(2 * pi) + 2 * sum(log(diag(U))) +
                       sum(backsolve(U, e, transpose = TRUE)^2))
  )
}
