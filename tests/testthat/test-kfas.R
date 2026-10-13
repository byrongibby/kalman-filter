models <- list(
  local_level = local_level(),
  local_trend = local_trend(),
  bivariate   = bivariate()
)

test_that("kfas returns every output with the documented dimensions", {
  mod <- bivariate()
  y <- simulate_y(mod, n = 25)
  out <- run_kfas(mod, y)

  n <- 25; p <- 2; m <- 3; q <- 2
  expect_named(out, c("v", "F", "att", "Ptt", "a", "P", "K", "r", "N",
                      "epshat", "V_eps", "etahat", "V_eta", "alphahat", "V"))
  expect_equal(dim(out$v), c(n, p))
  expect_equal(dim(out$F), c(p, n))
  expect_equal(dim(out$att), c(n, m))
  expect_equal(dim(out$Ptt), c(m, m, n))
  expect_equal(dim(out$a), c(n + 1, m))
  expect_equal(dim(out$P), c(m, m, n + 1))
  expect_equal(dim(out$K), c(m, p, n))
  expect_equal(dim(out$r), c(n + 1, m))
  expect_equal(dim(out$N), c(m, m, n + 1))
  expect_equal(dim(out$epshat), c(n, p))
  expect_equal(dim(out$V_eps), c(p, n))
  expect_equal(dim(out$etahat), c(n, q))
  expect_equal(dim(out$V_eta), c(q, q, n))
  expect_equal(dim(out$alphahat), c(n, m))
  expect_equal(dim(out$V), c(m, m, n))
})

test_that("kfas is deterministic (no reads of uninitialised memory)", {
  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    expect_identical(run_kfas(mod, y), run_kfas(mod, y), label = name)
  }
})

test_that("local level filter converges to its closed-form steady state", {
  # With H = Q = 1 the steady-state P solves P = P - P^2 / (P + 1) + 1,
  # giving P = (1 + sqrt(5)) / 2 and F = P + 1
  mod <- local_level()
  out <- run_kfas(mod, simulate_y(mod, n = 60))
  expect_equal(out$F[1, 60], (1 + sqrt(5)) / 2 + 1, tolerance = 1e-10)
})

test_that("filter output satisfies the Kalman recursions", {
  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    out <- run_kfas(mod, y)
    RQR <- mod$R %*% mod$Q %*% t(mod$R)

    expect_equal(out$a[1, ], mod$a1, label = name)
    expect_equal(out$P[, , 1], drop(mod$P1), label = name)
    for (t in seq_len(nrow(y))) {
      expect_equal(out$a[t + 1, ], drop(mod$T %*% out$att[t, ]), label = name)
      expect_equal(out$P[, , t + 1],
                   drop(mod$T %*% out$Ptt[, , t] %*% t(mod$T) + RQR),
                   label = name)
    }
  }
})

test_that("smoothed state at the final time equals the filtered state", {
  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    out <- run_kfas(mod, y)
    n <- nrow(y)

    expect_equal(out$r[n + 1, ], rep(0, ncol(mod$Z)), label = name)
    expect_equal(out$alphahat[n, ], out$att[n, ], label = name)
    expect_equal(out$V[, , n], out$Ptt[, , n], label = name)
  }
})

test_that("kfas matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    out <- run_kfas(mod, y)
    ref <- run_reference(mod, y)

    for (field in c("v", "F", "att", "Ptt", "a", "P", "alphahat", "V",
                    "epshat", "V_eps", "etahat", "V_eta")) {
      expect_same_values(out[[field]], ref[[field]])
    }
  }
})
