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

simulate_y <- function(mod, n = 40, seed = 1) {
  set.seed(seed)
  p <- nrow(mod$Z)
  m <- ncol(mod$Z)
  q <- ncol(mod$R)
  y <- matrix(0, n, p)
  alpha <- mod$a1 + drop(t(chol(mod$P1)) %*% rnorm(m))
  for (t in seq_len(n)) {
    y[t, ] <- mod$Z %*% alpha + rnorm(p, sd = sqrt(diag(mod$H)))
    alpha <- mod$T %*% alpha + mod$R %*% rnorm(q, sd = sqrt(diag(mod$Q)))
  }
  y
}

run_kfas <- function(mod, y) {
  kfas(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1)
}

# The multivariate routine in src/kfas.cpp is not exported
run_kfas_multivariate <- function(mod, y) {
  x <- .Call("kfas", t(y), mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1,
             mod$P1, PACKAGE = "kfilter")
  n <- nrow(y)
  m <- ncol(mod$Z)
  list(a = t(x[[5]]), att = t(x[[3]]), P = array(x[[6]], c(m, m, n + 1)),
       Ptt = array(x[[4]], c(m, m, n)), alphahat = t(x[[10]]),
       V = array(x[[11]], c(m, m, n)))
}

run_reference <- function(mod, y) {
  # SSModel only recognises SSMcustom by its bare name in the formula
  SSMcustom <- KFAS::SSMcustom
  ssm <- KFAS::SSModel(
    y ~ -1 + SSMcustom(Z = mod$Z, T = mod$T, R = mod$R, Q = mod$Q,
                       a1 = mod$a1, P1 = mod$P1),
    H = mod$H)
  KFAS::KFS(ssm, filtering = "state", smoothing = c("state", "disturbance"))
}

# Compare values only: KFAS returns ts objects and differently shaped arrays
expect_same_values <- function(object, expected, tolerance = 1e-8) {
  expect_equal(as.vector(object), as.vector(expected), tolerance = tolerance)
}
