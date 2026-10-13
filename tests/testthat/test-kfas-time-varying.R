matrices <- c("Z", "H", "T", "R", "Q")

test_that("kfas with time-varying matrices matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  # Each matrix varying on its own (mixing zero and non-zero strides),
  # then all of them together
  cases <- c(as.list(matrices), list(matrices))
  for (base in list(local_trend(), bivariate())) {
    for (which in cases) {
      mod <- time_varying(base, which)
      y <- simulate_y(mod)
      out <- run_kfas(mod, y)
      ref <- run_reference(mod, y)

      for (field in c("v", "F", "att", "Ptt", "a", "P", "alphahat", "V",
                      "epshat", "V_eps", "etahat", "V_eta")) {
        expect_same_values(out[[field]], ref[[field]])
      }
    }
  }
})

test_that("time-varying arrays of identical slices match the constant model", {
  mod <- bivariate()
  y <- simulate_y(mod)
  n <- nrow(y)

  stacked <- mod
  for (name in matrices) {
    stacked[[name]] <- array(mod[[name]], c(dim(mod[[name]]), n))
  }

  expect_equal(run_kfas(stacked, y), run_kfas(mod, y))
})

test_that("time-varying smoother satisfies its end conditions", {
  mod <- time_varying(bivariate(), matrices)
  y <- simulate_y(mod)
  out <- run_kfas(mod, y)
  n <- nrow(y)

  expect_equal(out$alphahat[n, ], out$att[n, ])
  expect_equal(out$V[, , n], out$Ptt[, , n])
})

test_that("kfas rejects system matrices of the wrong length", {
  mod <- bivariate()
  y <- simulate_y(mod)

  # Wrong number of time slices
  bad <- mod
  bad$Q <- array(mod$Q, c(dim(mod$Q), nrow(y) - 1))
  expect_error(run_kfas(bad, y), "Q must have 4 .* or 160 .* elements")

  # Wrong matrix size
  bad <- mod
  bad$H <- diag(3)
  expect_error(run_kfas(bad, y), "H must have 4 .* or 160 .* elements")
})
