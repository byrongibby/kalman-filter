all_matrices <- c("Z", "H", "T", "R", "Q")

# The precision sampler needs R Q R' non-singular: give every state of the
# bivariate model its own disturbance
full_rank <- function(mod) {
  mod$R <- diag(3)
  mod$Q <- diag(c(0.5, 0.3, 0.2))
  mod
}

run_sampler <- function(mod, y, nsim = 1, d = NULL, c = NULL) {
  precision_sampler(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1,
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

# Model, optional transformation of y (f) and intercepts (d, c)
cases <- list(
  local_level    = list(mod = local_level()),
  local_trend    = list(mod = local_trend()),
  bivariate      = list(mod = full_rank(bivariate())),
  time_varying   = list(mod = time_varying(full_rank(bivariate()),
                                           all_matrices)),
  missing        = list(mod = full_rank(bivariate()), f = missing_y),
  diffuse        = list(mod = make_diffuse(local_trend())),
  partial        = list(mod = make_diffuse(full_rank(bivariate()),
                                           c(TRUE, FALSE, TRUE)),
                        f = missing_y),
  full_H         = list(mod = full_H(full_rank(bivariate()))),
  full_H_missing = list(mod = make_diffuse(full_H(full_rank(bivariate()))),
                        f = missing_y),
  intercepts     = list(mod = make_diffuse(full_rank(bivariate())),
                        f = missing_y, d = d_tv, c = c_tv)
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
    sim <- run_sampler(case$mod, y, d = case$d, c = case$c)
    ref <- run_kfas_int(case$mod, y, case$d, case$c)
    expect_equal(sim$mean, ref$alphahat, tolerance = 1e-8, label = name)
  }
})

test_that("draws have the smoothed mean and variance", {
  nsim <- 20000
  for (name in c("bivariate", "full_H_missing", "intercepts")) {
    case <- cases[[name]]
    y <- case_data(case)
    set.seed(2)
    sim <- run_sampler(case$mod, y, nsim, case$d, case$c)
    ref <- run_kfas_int(case$mod, y, case$d, case$c)
    m <- ncol(ref$alphahat)

    # Sample means within 4.5 standard errors
    sd <- sqrt(t(apply(ref$V, 3, diag)))
    z <- (apply(sim$draws, c(1, 2), mean) - ref$alphahat) / (sd / sqrt(nsim))
    expect_lt(max(abs(z)), 4.5, label = paste(name, "mean"))

    # Sample covariance of each alpha_t, against V_t (standard error of a
    # variance about 1% at this nsim)
    for (t in c(1, 2, 20, n)) {
      S <- cov(t(matrix(sim$draws[t, , ], m)))
      scale <- sqrt(outer(diag(ref$V[, , t]), diag(ref$V[, , t])))
      expect_lt(max(abs(S - ref$V[, , t]) / scale), 0.05,
                label = paste(name, "V at", t))
    }
  }
})

test_that("draws are reproducible with set.seed and have the right shape", {
  mod <- full_rank(bivariate())
  y <- simulate_y(mod)
  set.seed(5)
  a <- run_sampler(mod, y, 3)
  set.seed(5)
  b <- run_sampler(mod, y, 3)
  expect_identical(a, b)
  expect_equal(dim(a$draws), c(n, 3, 3))
  expect_equal(dim(a$mean), c(n, 3))
  expect_false(identical(a$draws[, , 1], a$draws[, , 2]))
})

test_that("a single time step and a single state work", {
  mod <- local_level()
  y <- simulate_y(mod, n = 1)
  expect_equal(run_sampler(mod, y)$mean, run_kfas(mod, y)$alphahat)
})

test_that("models outside the sampler's requirements are rejected", {
  y <- simulate_y(bivariate())
  expect_error(run_sampler(bivariate(), y),
               "R Q R' is not positive definite at time 1")

  mod <- full_rank(bivariate())
  zero_H <- mod
  zero_H$H <- diag(c(1, 0))
  expect_error(run_sampler(zero_H, y), "H must be positive definite")

  singular_P1 <- mod
  singular_P1$P1[1, 1] <- 0
  expect_error(run_sampler(singular_P1, y), "P1 must be positive definite")

  # A diffuse state with no observations is not identified
  diffuse <- make_diffuse(local_level())
  expect_error(run_sampler(diffuse, matrix(NA_real_, n, 1)),
               "not identified")

  expect_error(run_sampler(mod, y, nsim = 0), "nsim must be a positive")
})
