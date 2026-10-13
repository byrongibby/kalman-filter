all_matrices <- c("Z", "H", "T", "R", "Q")

run_simsmooth <- function(mod, y, nsim = 1, d = NULL, c = NULL) {
  simulation_smoother(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1,
                      mod$P1inf, obsint = d, stateint = c, nsim = nsim)
}

run_kfas_int <- function(mod, y, d = NULL, c = NULL) {
  kfas(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1, mod$P1inf,
       obsint = d, stateint = c)
}

n <- 40
set.seed(10)
d_tv <- matrix(rnorm(2 * n, sd = 2), n, 2)
c_tv <- matrix(rnorm(3 * n), n, 3)
missing_y <- function(y) { y[1, 1] <- NA; y[5, ] <- NA; y[9, 2] <- NA; y }

# Local trend with a fixed slope: R Q R' is singular
fixed_slope <- function() {
  mod <- local_trend()
  mod$R <- matrix(c(1, 0), 2)
  mod$Q <- matrix(0.5)
  mod
}

# Series 2 with zero noise variance and no loading on the states at the
# first time steps: its observations there are uninformative (F = 0)
uninformative <- function() {
  mod <- time_varying(bivariate(), "Z")
  mod$Z[2, , 1:3] <- 0
  mod$H <- diag(c(1, 0))
  mod
}

# Model, optional transformation of y (f) and intercepts (d, c). The
# bivariate model has q = 2 < m = 3, so R Q R' is singular.
cases <- list(
  local_level    = list(mod = local_level()),
  bivariate      = list(mod = bivariate()),
  fixed_slope    = list(mod = make_diffuse(fixed_slope())),
  time_varying   = list(mod = time_varying(bivariate(), all_matrices)),
  missing        = list(mod = bivariate(), f = missing_y),
  uninformative  = list(mod = uninformative()),
  diffuse        = list(mod = make_diffuse(local_trend())),
  partial        = list(mod = make_diffuse(bivariate(), c(TRUE, FALSE, TRUE)),
                        f = missing_y),
  F_inf_zero     = list(mod = partial_no_loading(),
                        f = function(y) { y[1:2, 1] <- NA; y }),
  full_H_missing = list(mod = make_diffuse(full_H(bivariate())),
                        f = missing_y),
  intercepts     = list(mod = make_diffuse(bivariate()), f = missing_y,
                        d = d_tv, c = c_tv)
)

case_data <- function(case) {
  y <- simulate_y(case$mod) + 3
  if (!is.null(case$f)) y <- case$f(y)
  y
}

test_that("the posterior mean equals the smoothed state from kfas()", {
  for (name in names(cases)) {
    case <- cases[[name]]
    y <- case_data(case)
    sim <- run_simsmooth(case$mod, y, d = case$d, c = case$c)
    ref <- run_kfas_int(case$mod, y, case$d, case$c)
    expect_equal(sim$mean, ref$alphahat, tolerance = 1e-12, label = name)
  }
})

test_that("draws have the smoothed mean and variance", {
  nsim <- 20000
  for (name in names(cases)) {
    case <- cases[[name]]
    y <- case_data(case)
    set.seed(2)
    sim <- run_simsmooth(case$mod, y, nsim, case$d, case$c)
    ref <- run_kfas_int(case$mod, y, case$d, case$c)
    m <- ncol(ref$alphahat)

    # Sample means within 4.5 standard errors (elements with zero
    # posterior variance must equal the mean)
    sd <- matrix(sqrt(pmax(apply(ref$V, 3, diag), 0)), n, m, byrow = TRUE)
    err <- abs(apply(sim$draws, c(1, 2), mean) - ref$alphahat)
    expect_true(all(err <= 4.5 * sd / sqrt(nsim) + 1e-8),
                label = paste(name, "mean"))

    # Sample covariance of each alpha_t, against V_t (standard error of a
    # variance about 1% at this nsim)
    for (t in c(1, 2, 3, 20, n)) {
      S <- cov(t(matrix(sim$draws[t, , ], m)))
      V_t <- matrix(ref$V[, , t], m)
      scale <- sqrt(outer(diag(V_t), diag(V_t))) + 1e-12
      expect_lt(max(abs(S - V_t) / scale), 0.05,
                label = paste(name, "V at", t))
    }
  }
})

test_that("draws do not depend on inita in the diffuse directions", {
  for (name in c("diffuse", "partial", "F_inf_zero", "intercepts")) {
    case <- cases[[name]]
    y <- case_data(case)
    shifted <- case$mod
    diffuse <- diag(shifted$P1inf) > 0
    shifted$a1[diffuse] <- shifted$a1[diffuse] + 100
    set.seed(3)
    a <- run_simsmooth(case$mod, y, 5, case$d, case$c)
    set.seed(3)
    b <- run_simsmooth(shifted, y, 5, case$d, case$c)
    expect_equal(b, a, tolerance = 1e-9, label = name)
  }
})

test_that("draws agree in distribution with the precision sampler", {
  mod <- make_diffuse(time_varying(bivariate(), all_matrices),
                      c(TRUE, FALSE, TRUE))
  mod$R <- diag(3)
  mod$Q <- diag(c(0.5, 0.3, 0.2))
  y <- missing_y(simulate_y(mod))
  nsim <- 20000
  set.seed(4)
  a <- run_simsmooth(mod, y, nsim, d_tv, c_tv)$draws
  b <- precision_sampler(
    y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1, mod$P1inf,
    obsint = d_tv, stateint = c_tv, nsim = nsim)$draws

  # Covariance across time steps, which kfas() does not report: alpha_t
  # against alpha_t+1 for the first state, within 4.5 standard errors of
  # the difference between the two estimates
  for (t in c(1, 10, n - 1)) {
    x1 <- b[t, 1, ]
    x2 <- b[t + 1, 1, ]
    se <- sqrt(2 * (var(x1) * var(x2) + cov(x1, x2)^2) / nsim)
    expect_lt(abs(cov(a[t, 1, ], a[t + 1, 1, ]) - cov(x1, x2)), 4.5 * se,
              label = paste("lag-1 covariance at", t))
  }
})

test_that("draws are reproducible with set.seed and have the right shape", {
  mod <- bivariate()
  y <- simulate_y(mod)
  set.seed(5)
  a <- run_simsmooth(mod, y, 3)
  set.seed(5)
  b <- run_simsmooth(mod, y, 3)
  expect_identical(a, b)
  expect_equal(dim(a$draws), c(n, 3, 3))
  expect_equal(dim(a$mean), c(n, 3))
  expect_false(identical(a$draws[, , 1], a$draws[, , 2]))
})

test_that("a single time step works", {
  mod <- local_level()
  y <- simulate_y(mod, n = 1)
  sim <- run_simsmooth(mod, y, 2)
  expect_equal(sim$mean, run_kfas(mod, y)$alphahat)
  expect_equal(dim(sim$draws), c(1, 1, 2))
})

test_that("invalid inputs are rejected", {
  mod <- bivariate()
  y <- simulate_y(mod)
  expect_error(run_simsmooth(mod, y, nsim = 0), "nsim must be a positive")
  bad <- mod
  bad$Q <- diag(c(0.5, -0.3))
  expect_error(run_simsmooth(bad, y), "Q must be positive semi-definite")
})
