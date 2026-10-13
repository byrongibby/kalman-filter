# Hyperplane-truncated draws from precision_sampler() (Cong, Chen and Zhou,
# 2017): the posterior of the states conditional on constraints A alpha = r

# The precision sampler needs R Q R' non-singular
full_rank <- function(mod) {
  mod$R <- diag(3)
  mod$Q <- diag(c(0.5, 0.3, 0.2))
  mod
}

run_constrained <- function(mod, y, A, r, nsim = 1) {
  precision_sampler(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1,
                    mod$P1inf, nsim = nsim, constraints = A, rhs = r)
}

n <- 40
m <- 3

# Constraint on as.vector(alpha): coefficient coef on state i at time t
coef_row <- function(t, i, coef = 1) {
  row <- matrix(0, n, m)
  row[cbind(t, i)] <- coef
  as.vector(row)
}

# Cross-time constraints: the first state averages to given values over
# each block of ten time steps, and states 2 and 3 sum to 1 at time 7
cross_time <- rbind(
  t(sapply(0:3, function(b) coef_row(b * 10 + 1:10, 1, 0.1))),
  coef_row(c(7, 7), 2:3)
)
cross_rhs <- c(1, -1, 0.5, 2, 1)

test_that("every draw satisfies the constraints", {
  mod <- full_rank(bivariate())
  y <- simulate_y(mod)
  sim <- run_constrained(mod, y, cross_time, cross_rhs, 50)
  ax <- cross_time %*% matrix(sim$draws, n * m)
  expect_equal(ax, matrix(cross_rhs, 5, 50), tolerance = 1e-10)
  expect_equal(drop(cross_time %*% as.vector(sim$mean)), cross_rhs,
               tolerance = 1e-10)
})

test_that("cross-time constraints match the exact conditional distribution", {
  mod <- full_rank(time_varying(bivariate(), c("Z", "T", "Q")))
  y <- simulate_y(mod)
  y[c(3, 15), 1] <- NA
  ex <- exact_posterior(mod, y)

  # Conditioning the dense posterior N(mu, S) on A alpha = r
  mu <- ex$stacked_mean
  S <- ex$stacked_var
  SA <- S %*% t(cross_time)
  G <- solve(cross_time %*% SA)
  mu_c <- mu + drop(SA %*% G %*% (cross_rhs - drop(cross_time %*% mu)))
  S_c <- S - SA %*% G %*% t(SA)

  nsim <- 20000
  set.seed(1)
  sim <- run_constrained(mod, y, cross_time, cross_rhs, nsim)
  expect_equal(as.vector(sim$mean), mu_c, tolerance = 1e-8)

  # Sample moments of a selection of elements, including pairs within one
  # averaging block and across time steps
  x <- matrix(sim$draws, n * m)
  idx <- c(1, 5, 10, 11, 25, n + 7, 2 * n + 7, 3 * n)
  sd <- sqrt(diag(S_c)[idx])
  z <- (rowMeans(x[idx, ]) - mu_c[idx]) / (sd / sqrt(nsim))
  expect_lt(max(abs(z)), 4.5)
  C <- cov(t(x[idx, ]))
  expect_lt(max(abs(C - S_c[idx, idx]) / outer(sd, sd)), 0.05)
})

test_that("per-time constraints match kfas() with a noise-free observation", {
  # alpha_t,1 + alpha_t,2 = 1 at three time steps, the same as observing a
  # third series y_t,3 = alpha_t,1 + alpha_t,2 with zero noise there
  times <- c(1, 12, 30)
  A <- t(sapply(times, function(t) coef_row(c(t, t), 1:2)))
  r <- rep(1, 3)

  for (diffuse in c(FALSE, TRUE)) {
    mod <- full_rank(bivariate())
    if (diffuse) mod <- make_diffuse(mod, c(TRUE, FALSE, TRUE))
    y <- simulate_y(mod)
    y[5, 2] <- NA

    aug <- mod
    aug$Z <- rbind(mod$Z, c(1, 1, 0))
    aug$H <- diag(c(diag(mod$H), 0))
    y3 <- rep(NA_real_, n)
    y3[times] <- 1
    ref <- run_kfas(aug, cbind(y, y3))

    nsim <- 20000
    set.seed(2)
    sim <- run_constrained(mod, y, A, r, nsim)
    expect_equal(sim$mean, ref$alphahat, tolerance = 1e-8,
                 label = paste("mean, diffuse =", diffuse))
    for (t in c(1, 2, 12, n)) {
      S <- cov(t(matrix(sim$draws[t, , ], m)))
      V_t <- ref$V[, , t]
      expect_lt(max(abs(S - V_t) / sqrt(outer(diag(V_t), diag(V_t)) + 1e-12)),
                0.05, label = paste("V at", t, "diffuse =", diffuse))
    }
  }
})

test_that("constraints can be a matrix, an array or a single vector", {
  mod <- full_rank(bivariate())
  y <- simulate_y(mod)
  set.seed(3)
  a <- run_constrained(mod, y, cross_time, cross_rhs, 2)
  set.seed(3)
  b <- run_constrained(mod, y, array(cross_time, c(5, n, m)), cross_rhs, 2)
  expect_identical(a, b)

  set.seed(3)
  one <- run_constrained(mod, y, cross_time[1, ], 1, 2)
  set.seed(3)
  one_matrix <- run_constrained(mod, y, cross_time[1, , drop = FALSE], 1, 2)
  expect_identical(one, one_matrix)

  # Without rhs, the constraints are homogeneous
  zero <- run_constrained(mod, y, cross_time, NULL)
  expect_equal(drop(cross_time %*% as.vector(zero$mean)), rep(0, 5),
               tolerance = 1e-10)
})

test_that("draws without constraints are unchanged", {
  mod <- full_rank(bivariate())
  y <- simulate_y(mod)
  set.seed(4)
  a <- precision_sampler(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1,
                         mod$P1, nsim = 3)
  set.seed(4)
  b <- run_constrained(mod, y, NULL, NULL, 3)
  expect_identical(a, b)
})

test_that("invalid constraints are rejected", {
  mod <- full_rank(bivariate())
  y <- simulate_y(mod)
  expect_error(run_constrained(mod, y, rbind(cross_time, cross_time[1, ]),
                               c(cross_rhs, 1)),
               "constraints must be linearly independent")
  expect_error(run_constrained(mod, y, cross_time[, -1], cross_rhs),
               "constraints must have n \\* m = 120 columns")
  expect_error(run_constrained(mod, y, cross_time, 1:2),
               "rhs must have one element per constraint \\(5\\)")
  expect_error(run_constrained(mod, y, NULL, 1), "rhs is given without")
})
