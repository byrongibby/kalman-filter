all_matrices <- c("Z", "H", "T", "R", "Q")
models <- list(
  local_level  = local_level(),
  local_trend  = local_trend(),
  bivariate    = bivariate(),
  time_varying = time_varying(bivariate(), all_matrices),
  full_H       = full_H(bivariate()),
  full_H_tv    = time_varying(full_H(bivariate()), all_matrices)
)

test_that("logLik of a single observation is the normal density", {
  mod <- local_level()
  y <- simulate_y(mod, n = 1)
  expect_equal(run_kfas(mod, y)$logLik,
               dnorm(y[1], mod$a1, sqrt(mod$P1[1] + mod$H[1]), log = TRUE))

  # Bivariate with non-diagonal H: y_1 ~ N(Z a1, Z P1 Z' + H)
  mod <- full_H(bivariate())
  y <- simulate_y(mod, n = 1)
  S <- mod$Z %*% mod$P1 %*% t(mod$Z) + mod$H
  e <- drop(y) - drop(mod$Z %*% mod$a1)
  expected <- -0.5 * (2 * log(2 * pi) + log(det(S)) +
                        drop(e %*% solve(S, e)))
  expect_equal(run_kfas(mod, y)$logLik, expected)
  expect_equal(run_kfas_multivariate(mod, y)$logLik, expected)
})

test_that("logLik equals the exact joint density of all observations", {
  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    expected <- exact_posterior(mod, y)$logLik

    expect_equal(run_kfas(mod, y)$logLik, expected, tolerance = 1e-10,
                 label = paste(name, "univariate"))
    expect_equal(run_kfas_multivariate(mod, y)$logLik, expected,
                 tolerance = 1e-10, label = paste(name, "multivariate"))
  }
})

test_that("logLik matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    ref <- run_reference(mod, y)$logLik

    expect_equal(run_kfas(mod, y)$logLik, ref, tolerance = 1e-10,
                 label = paste(name, "univariate"))
    expect_equal(run_kfas_multivariate(mod, y)$logLik, ref,
                 tolerance = 1e-10, label = paste(name, "multivariate"))
  }
})

test_that("uninformative observations do not contribute to logLik", {
  mod <- bivariate()
  mod$Z[2, ] <- 0
  mod$H[2, 2] <- 0
  y <- simulate_y(mod)

  single <- mod
  single$Z <- mod$Z[1, , drop = FALSE]
  single$H <- mod$H[1, 1, drop = FALSE]

  expect_equal(run_kfas(mod, y)$logLik,
               run_kfas(single, y[, 1, drop = FALSE])$logLik)
})

test_that("logLik does not depend on the smoother and disturbance flags", {
  mod <- full_H(bivariate())
  y <- simulate_y(mod)
  full <- run_kfas(mod, y)$logLik
  call <- function(...) {
    kfas(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1, ...)$logLik
  }

  expect_identical(call(smoother = FALSE, disturbances = FALSE), full)
  expect_identical(
    run_kfas_multivariate(mod, y, smoother = FALSE,
                          disturbances = FALSE)$logLik,
    run_kfas_multivariate(mod, y)$logLik)
})
