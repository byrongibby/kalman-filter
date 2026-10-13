all_matrices <- c("Z", "H", "T", "R", "Q")

# Intercepts d_t and c_t, checked against the equivalent model without
# intercepts: a constant state alpha_0 = 1 is appended, with Z* = [Z, d] and
# T* = [T, c; 0, 1]. The original states, and every output for them, are
# unchanged by the augmentation.

# Slice t of an intercept given as a vector or a time-by-element matrix
at_int <- function(x, t) if (is.matrix(x)) x[t, ] else x

augment <- function(mod, d, c, n = 40) {
  m <- dim(mod$R)[1]
  p <- dim(as.array(mod$Z))[1]
  q <- dim(mod$R)[2]
  Z <- array(vapply(seq_len(n), function(t) {
    cbind(matrix(at(mod$Z, t), p), at_int(d, t))
  }, matrix(0, p, m + 1)), c(p, m + 1, n))
  T <- array(vapply(seq_len(n), function(t) {
    rbind(cbind(at(mod$T, t), at_int(c, t)), c(rep(0, m), 1))
  }, matrix(0, m + 1, m + 1)), c(m + 1, m + 1, n))
  R <- if (length(dim(mod$R)) == 3) {
    array(vapply(seq_len(n), function(t) rbind(mod$R[, , t], 0),
                 matrix(0, m + 1, q)), c(m + 1, q, n))
  } else {
    rbind(mod$R, 0)
  }
  pad <- function(P) rbind(cbind(P, 0), 0)
  P1inf <- if (is.null(mod$P1inf)) NULL else pad(mod$P1inf)
  list(Z = Z, H = mod$H, T = T, R = R, Q = mod$Q, a1 = c(mod$a1, 1),
       P1 = pad(mod$P1), P1inf = P1inf)
}

run_intercepts <- function(mod, y, d, c) {
  kfas(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1, mod$P1inf,
       obsint = d, stateint = c)
}

# Restrict the outputs of the augmented model to the original m states
drop_constant <- function(out, m) {
  s <- seq_len(m)
  out$att <- out$att[, s, drop = FALSE]
  out$Ptt <- out$Ptt[s, s, , drop = FALSE]
  out$a <- out$a[, s, drop = FALSE]
  out$P <- out$P[s, s, , drop = FALSE]
  out$K <- out$K[s, , , drop = FALSE]
  out$r <- out$r[, s, drop = FALSE]
  out$N <- out$N[s, s, , drop = FALSE]
  out$alphahat <- out$alphahat[, s, drop = FALSE]
  out$V <- out$V[s, s, , drop = FALSE]
  out$Pinf <- out$Pinf[s, s, , drop = FALSE]
  out
}

n <- 40
set.seed(10)
d_tv <- matrix(rnorm(2 * n, sd = 2), n, 2)
c_tv <- matrix(rnorm(3 * n), n, 3)

cases <- list(
  local_level   = list(local_level(), 5, 0.3, NULL),
  trend_diffuse = list(make_diffuse(local_trend()), 2, c(0.1, -0.05), NULL),
  bivariate     = list(bivariate(), c(1, -2), c(0.5, 0, -0.3), NULL),
  time_varying  = list(time_varying(bivariate(), all_matrices), d_tv, c_tv,
                       NULL),
  d_only        = list(bivariate(), d_tv, NULL, NULL),
  c_only        = list(bivariate(), NULL, c_tv, NULL),
  diffuse       = list(make_diffuse(bivariate(), c(TRUE, FALSE, TRUE)),
                       d_tv, c_tv, NULL),
  missing       = list(make_diffuse(bivariate()), d_tv, c_tv, function(y) {
                    y[1, 1] <- NA; y[5, ] <- NA; y[9, 2] <- NA; y }),
  full_H        = list(full_H(bivariate()), d_tv, c_tv, NULL),
  full_H_missing = list(make_diffuse(full_H(bivariate())), d_tv, c_tv,
                        function(y) { y[1, 1] <- NA; y[3, 2] <- NA; y })
)

case_data <- function(case) {
  y <- simulate_y(case[[1]]) + 3
  if (!is.null(case[[4]])) y <- case[[4]](y)
  y
}

test_that("intercepts match the model with a constant state appended", {
  for (name in names(cases)) {
    case <- cases[[name]]
    mod <- case[[1]]
    y <- case_data(case)
    m <- dim(mod$R)[1]
    d <- if (is.null(case[[2]])) 0 else case[[2]]
    c <- if (is.null(case[[3]])) rep(0, m) else case[[3]]

    out <- run_intercepts(mod, y, case[[2]], case[[3]])
    ref <- drop_constant(run_kfas(augment(mod, d, c), y), m)

    expect_equal(out$d, ref$d, label = paste(name, "d"))
    expect_equal(out$j, ref$j, label = paste(name, "j"))
    expect_equal(out$logLik, ref$logLik, tolerance = 1e-10,
                 label = paste(name, "logLik"))
    for (field in setdiff(names(out), c("d", "j", "logLik"))) {
      expect_equal(out[[field]], ref[[field]], tolerance = 1e-8,
                   label = paste(name, field))
    }
  }
})

test_that("kfas_loglik with intercepts is identical to kfas()", {
  for (name in names(cases)) {
    case <- cases[[name]]
    mod <- case[[1]]
    y <- case_data(case)
    expect_identical(
      kfas_loglik(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1,
                  mod$P1inf, obsint = case[[2]], stateint = case[[3]]),
      run_intercepts(mod, y, case[[2]], case[[3]])$logLik, label = name)
  }
})

test_that("the observation intercept is equivalent to subtracting it from y", {
  mod <- bivariate()
  y <- simulate_y(mod)
  out <- run_intercepts(mod, y + d_tv, d_tv, NULL)
  expect_equal(out, run_kfas(mod, y))
})

test_that("zero intercepts give exactly the result without intercepts", {
  mod <- make_diffuse(full_H(bivariate()))
  y <- simulate_y(mod)
  y[4, 1] <- NA
  expect_identical(run_intercepts(mod, y, c(0, 0), rep(0, 3)),
                   run_kfas(mod, y))
  expect_identical(run_intercepts(mod, y, matrix(0, n, 2), matrix(0, n, 3)),
                   run_kfas(mod, y))
})

test_that("a constant intercept can be given as a vector or as rows", {
  mod <- bivariate()
  y <- simulate_y(mod)
  d <- c(1, -2)
  c <- c(0.5, 0, -0.3)
  expect_identical(
    run_intercepts(mod, y, matrix(d, n, 2, byrow = TRUE),
                   matrix(c, n, 3, byrow = TRUE)),
    run_intercepts(mod, y, d, c))
})

test_that("intercepts of the wrong size are rejected", {
  mod <- bivariate()
  y <- simulate_y(mod)
  expect_error(run_intercepts(mod, y, 1, NULL),
               "d must have 2 .* or 80 .* elements, not 1")
  expect_error(run_intercepts(mod, y, NULL, c(1, 2)),
               "c must have 3 .* or 120 .* elements, not 2")
  expect_error(run_intercepts(mod, y, "a", NULL), "obsint must be numeric")
})
