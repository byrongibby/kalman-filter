# Non-diagonal H is handled by filtering the model transformed with the
# Cholesky factor H = L L'. States and eta disturbances are invariant to
# the transform; v, F and K are reported in the Cholesky-transformed scale;
# epsilon disturbances are reported in the original scale.

# KFAS reports v, F, epshat and V_eps in the scale of its LDL transform
# H = C D C'. Since L = C D^(1/2), the Cholesky scale divides v by
# sqrt(D) and F by D; the original-scale disturbances follow from the
# smoothed states.
expected_from_reference <- function(mod, y, ref) {
  n <- nrow(y)
  p <- ncol(y)
  v <- matrix(0, n, p)
  F <- matrix(0, p, n)
  epshat <- matrix(0, n, p)
  V_eps <- matrix(0, p, n)
  for (t in seq_len(n)) {
    D <- diag(t(chol(at(mod$H, t))))^2
    Z <- at(mod$Z, t)
    v[t, ] <- ref$v[t, ] / sqrt(D)
    F[, t] <- ref$F[, t] / D
    epshat[t, ] <- y[t, ] - Z %*% ref$alphahat[t, ]
    V_eps[, t] <- diag(Z %*% ref$V[, , t] %*% t(Z))
  }
  list(v = v, F = F, epshat = epshat, V_eps = V_eps)
}

expect_matches_reference <- function(mod, y) {
  out <- run_kfas(mod, y)
  ref <- run_reference(mod, y)
  expected <- expected_from_reference(mod, y, ref)

  for (field in c("att", "Ptt", "a", "P", "alphahat", "V", "etahat",
                  "V_eta")) {
    expect_same_values(out[[field]], ref[[field]])
  }
  for (field in names(expected)) {
    expect_same_values(out[[field]], expected[[field]])
  }
}

test_that("kfas with non-diagonal H matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  mod <- full_H(bivariate())
  expect_matches_reference(mod, simulate_y(mod))
})

test_that("kfas with time-varying non-diagonal H matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  for (which in list("H", c("Z", "H", "T", "R", "Q"))) {
    mod <- time_varying(full_H(bivariate()), which)
    expect_matches_reference(mod, simulate_y(mod))
  }
})

test_that("diagonal H gives the same result whether or not it is transformed", {
  # A diagonal H takes the untransformed path; a tiny off-diagonal forces
  # the Cholesky path, which should agree to within that perturbation
  mod <- bivariate()
  y <- simulate_y(mod)
  perturbed <- mod
  perturbed$H[1, 2] <- perturbed$H[2, 1] <- 1e-12

  out <- run_kfas(mod, y)
  out_chol <- run_kfas(perturbed, y)
  for (field in c("a", "P", "alphahat", "V", "epshat", "V_eps", "etahat")) {
    expect_equal(out_chol[[field]], out[[field]], tolerance = 1e-8)
  }
})

test_that("kfas rejects H that is not positive definite", {
  mod <- bivariate()
  mod$H <- matrix(c(1, 2, 2, 1), 2)
  expect_error(run_kfas(mod, simulate_y(bivariate())),
               "H is not positive definite at time 1")
})
