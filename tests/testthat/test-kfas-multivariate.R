# src/kfas.cpp (multivariate treatment) is not exported, but is tested so it
# stays at feature parity with the univariate kfas0 routine behind kfas()

all_matrices <- c("Z", "H", "T", "R", "Q")
models <- list(
  local_level  = local_level(),
  local_trend  = local_trend(),
  bivariate    = bivariate(),
  time_varying = time_varying(bivariate(), all_matrices),
  full_H       = full_H(bivariate()),
  full_H_tv    = time_varying(full_H(bivariate()), all_matrices)
)

# Outputs that do not depend on the treatment. v, F and K do: they are
# defined per series in the univariate treatment and per vector here.
shared_fields <- c("a", "P", "att", "Ptt", "r", "N", "alphahat", "V",
                   "epshat", "V_eps", "etahat", "V_eta")

test_that("multivariate routine is deterministic", {
  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    expect_identical(run_kfas_multivariate(mod, y),
                     run_kfas_multivariate(mod, y), label = name)
  }
})

test_that("multivariate and univariate treatments agree", {
  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    multi <- run_kfas_multivariate(mod, y)
    uni <- run_kfas(mod, y)

    for (field in shared_fields) {
      expect_equal(multi[[field]], uni[[field]], tolerance = 1e-8,
                   label = paste(name, field))
    }
  }
})

test_that("multivariate filter outputs satisfy their definitions", {
  for (name in c("bivariate", "full_H_tv")) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    out <- run_kfas_multivariate(mod, y)

    for (t in seq_len(nrow(y))) {
      Z <- at(mod$Z, t)
      P <- out$P[, , t]
      expect_equal(out$v[t, ], drop(y[t, ] - Z %*% out$a[t, ]))
      expect_equal(out$F[, , t], Z %*% P %*% t(Z) + at(mod$H, t))
      expect_equal(out$K[, , t], P %*% t(Z))
    }
  }
})

test_that("multivariate routine matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  # KFAS reports epsilon disturbances in an LDL-transformed scale for
  # non-diagonal H, so compare those only for diagonal H
  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    multi <- run_kfas_multivariate(mod, y)
    ref <- run_reference(mod, y)

    fields <- c("a", "P", "att", "Ptt", "alphahat", "V", "etahat", "V_eta")
    if (!startsWith(name, "full_H")) fields <- c(fields, "epshat", "V_eps")
    for (field in fields) {
      expect_same_values(multi[[field]], ref[[field]])
    }
  }
})

test_that("multivariate outputs that are not requested are NA", {
  mod <- bivariate()
  y <- simulate_y(mod)
  full <- run_kfas_multivariate(mod, y)
  filter_fields <- c("v", "F", "att", "Ptt", "a", "P", "K")

  out <- run_kfas_multivariate(mod, y, smoother = FALSE)
  expect_equal(out[c(filter_fields, "r", "N", "epshat", "V_eps")],
               full[c(filter_fields, "r", "N", "epshat", "V_eps")])
  expect_true(all(is.na(unlist(out[c("alphahat", "V")]))))

  out <- run_kfas_multivariate(mod, y, smoother = FALSE,
                               disturbances = FALSE)
  expect_equal(out[filter_fields], full[filter_fields])
  expect_true(all(is.na(unlist(out[c("r", "N", "alphahat", "V", "epshat",
                                     "V_eps", "etahat", "V_eta")]))))
})

test_that("multivariate routine rejects bad inputs and singular F", {
  mod <- bivariate()
  y <- simulate_y(mod)

  bad <- mod
  bad$Q <- array(mod$Q, c(dim(mod$Q), nrow(y) - 1))
  expect_error(run_kfas_multivariate(bad, y), "Q must have 4 .* or 160")

  bad <- mod
  bad$a1 <- 0
  expect_error(run_kfas_multivariate(bad, y), "a1 must have 3 elements")

  # A series with no loading and no noise makes F singular
  bad <- mod
  bad$Z[2, ] <- 0
  bad$H[2, 2] <- 0
  expect_error(run_kfas_multivariate(bad, y),
               "F is not positive definite at time 1")
})
