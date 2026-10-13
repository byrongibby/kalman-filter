all_matrices <- c("Z", "H", "T", "R", "Q")

# Partially diffuse model where series 2 does not load on the diffuse state,
# so F_inf = 0 for it while the diffuse phase continues
partial_no_loading <- function() {
  mod <- make_diffuse(bivariate(), c(TRUE, FALSE, FALSE))
  mod$Z[2, 1] <- 0
  mod
}

cases <- list(
  local_level      = list(make_diffuse(local_level()), NULL),
  local_trend      = list(make_diffuse(local_trend()), NULL),
  bivariate        = list(make_diffuse(bivariate()), NULL),
  partial          = list(make_diffuse(bivariate(), c(TRUE, FALSE, FALSE)), NULL),
  missing          = list(make_diffuse(bivariate()), function(y) {
                       y[1, 1] <- NA; y[2, ] <- NA; y }),
  first_missing    = list(make_diffuse(local_trend()), function(y) {
                       y[1] <- NA; y }),
  F_inf_zero       = list(partial_no_loading(), function(y) {
                       y[1:2, 1] <- NA; y }),
  time_varying     = list(make_diffuse(time_varying(bivariate(), all_matrices)), NULL),
  full_H           = list(make_diffuse(full_H(bivariate())), NULL),
  full_H_missing   = list(make_diffuse(full_H(bivariate())), function(y) {
                       y[1, 1] <- NA; y[3, 2] <- NA; y })
)

case_data <- function(case) {
  y <- simulate_y(case[[1]])
  if (!is.null(case[[2]])) y <- case[[2]](y)
  y
}

test_that("diffuse filter and smoother match KFAS::KFS", {
  skip_if_not_installed("KFAS")

  for (name in names(cases)) {
    mod <- cases[[name]][[1]]
    y <- case_data(cases[[name]])
    out <- run_kfas(mod, y)
    ref <- run_reference(mod, y)

    expect_equal(out$d, ref$d, label = paste(name, "d"))
    expect_equal(out$j, ref$j, label = paste(name, "j"))
    expect_equal(out$logLik, ref$logLik, tolerance = 1e-10,
                 label = paste(name, "logLik"))

    fields <- c("a", "P", "att", "Ptt", "alphahat", "V", "etahat", "V_eta")
    # v, F and the epsilon disturbances are on a different scale from
    # KFAS's for non-diagonal H
    if (!startsWith(name, "full_H")) {
      fields <- c(fields, "v", "F", "epshat", "V_eps", "Pinf", "Finf")
    }
    for (field in fields) {
      expect_same_values(out[[field]], ref[[field]])
    }
  }
})

test_that("diffuse epsilon disturbances for non-diagonal H equal y - Z alphahat", {
  for (name in c("full_H", "full_H_missing")) {
    mod <- cases[[name]][[1]]
    y <- case_data(cases[[name]])
    out <- run_kfas(mod, y)
    expected <- y - t(mod$Z %*% t(out$alphahat))
    observed <- !is.na(y)
    expect_equal(out$epshat[observed], expected[observed], tolerance = 1e-10)
  }
})

test_that("kfas_loglik with a diffuse initial state is identical to kfas()", {
  for (name in names(cases)) {
    mod <- cases[[name]][[1]]
    y <- case_data(cases[[name]])
    expect_identical(
      kfas_loglik(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1,
                  mod$P1inf),
      run_kfas(mod, y)$logLik, label = name)
  }
})

test_that("exact diffuse results are the limit of a large initial variance", {
  # P1 = P* + kappa P_inf with large kappa approximates the exact diffuse
  # filter, with error of order 1 / kappa
  for (name in c("local_trend", "partial", "missing")) {
    mod <- cases[[name]][[1]]
    y <- case_data(cases[[name]])
    exact <- run_kfas(mod, y)

    errors <- sapply(c(1e4, 1e6), function(kappa) {
      approx <- mod
      approx$P1 <- mod$P1 + kappa * mod$P1inf
      approx$P1inf <- NULL
      out <- run_kfas(approx, y)
      max(abs(out$alphahat - exact$alphahat))
    })
    # O(1 / kappa): 100 times larger kappa, about 100 times smaller error
    expect_lt(errors[2], 1e-3, label = paste(name, "error at kappa = 1e6"))
    expect_lt(errors[2], errors[1] / 50, label = paste(name, "convergence"))
  }
})

test_that("no diffuse elements gives the non-diffuse result", {
  mod <- bivariate()
  y <- simulate_y(mod)
  plain <- run_kfas(mod, y)
  zero <- mod
  zero$P1inf <- matrix(0, 3, 3)

  expect_identical(run_kfas(zero, y), plain)
  expect_equal(plain$d, 0L)
  expect_equal(dim(plain$Pinf), c(3, 3, 0))
})

test_that("initpinf of the wrong size is rejected", {
  mod <- bivariate()
  y <- simulate_y(mod)
  mod$P1inf <- diag(2)
  expect_error(run_kfas(mod, y), "P1inf must have 9 elements, not 4")
})
