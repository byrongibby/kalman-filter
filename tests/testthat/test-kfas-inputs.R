test_that("integer inputs give the same result as doubles", {
  mod <- local_level()
  y <- round(simulate_y(mod) * 10)
  int <- lapply(mod, function(x) { storage.mode(x) <- "integer"; x })
  y_int <- y
  storage.mode(y_int) <- "integer"

  expect_equal(run_kfas(int, y_int), run_kfas(mod, y))
})

test_that("kfas rejects malformed inputs", {
  mod <- local_trend()
  y <- simulate_y(mod)

  expect_error(run_kfas(mod, as.vector(y)), "obs must be an n x p matrix")

  bad <- mod
  bad$Q <- matrix("a", 2, 2)
  expect_error(run_kfas(bad, y), "must be numeric")

  bad <- mod
  bad$a1 <- 0
  expect_error(run_kfas(bad, y), "a1 must have 2 elements, not 1")

  bad <- mod
  bad$P1 <- diag(3)
  expect_error(run_kfas(bad, y), "P1 must have 4 elements, not 9")
})

test_that("observations with F = 0 are skipped instead of producing NaN", {
  # A series with no loading and no noise carries no information, so the
  # result must equal filtering the informative series alone
  mod <- bivariate()
  mod$Z[2, ] <- 0
  mod$H[2, 2] <- 0
  y <- simulate_y(mod)
  out <- run_kfas(mod, y)

  single <- mod
  single$Z <- mod$Z[1, , drop = FALSE]
  single$H <- mod$H[1, 1, drop = FALSE]
  ref <- run_kfas(single, y[, 1, drop = FALSE])

  expect_false(anyNA(out[c("att", "Ptt", "a", "P", "alphahat", "V")]))
  for (field in c("att", "Ptt", "a", "P", "alphahat", "V", "etahat",
                  "V_eta")) {
    expect_equal(out[[field]], ref[[field]], tolerance = 1e-10)
  }
  expect_equal(out$K[, 2, ], array(0, c(3, nrow(y))))
})

test_that("a degenerate first observation (H = 0, P1 = 0) matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  mod <- local_level()
  mod$H <- matrix(0)
  mod$P1 <- matrix(0)
  y <- simulate_y(local_level())
  out <- run_kfas(mod, y)
  ref <- run_reference(mod, y)

  expect_false(anyNA(out[c("att", "a", "alphahat", "V")]))
  for (field in c("att", "Ptt", "a", "P", "alphahat", "V")) {
    expect_same_values(out[[field]], ref[[field]])
  }
})

test_that("the F = 0 test is invariant to the scale of the data", {
  mod <- bivariate()
  y <- simulate_y(mod)
  out <- run_kfas(mod, y)

  # Rescale so every variance is around 1e-12
  s <- 1e-6
  small <- mod
  small$H <- mod$H * s^2
  small$Q <- mod$Q * s^2
  small$P1 <- mod$P1 * s^2
  small$a1 <- mod$a1 * s
  out_small <- run_kfas(small, y * s)

  expect_equal(out_small$alphahat, out$alphahat * s, tolerance = 1e-10)
  expect_equal(out_small$V, out$V * s^2, tolerance = 1e-10)
})

test_that("outputs that are not requested are NA and the rest unchanged", {
  mod <- bivariate()
  y <- simulate_y(mod)
  full <- run_kfas(mod, y)
  filter_fields <- c("v", "F", "att", "Ptt", "a", "P", "K")
  smoother_fields <- c("alphahat", "V")
  disturbance_fields <- c("epshat", "V_eps", "etahat", "V_eta")

  call <- function(mod, ...) {
    kfas(y, mod$Z, mod$H, mod$T, mod$R, mod$Q, mod$a1, mod$P1, ...)
  }

  out <- call(mod, smoother = FALSE)
  expect_equal(out[c(filter_fields, "r", "N", disturbance_fields)],
               full[c(filter_fields, "r", "N", disturbance_fields)])
  expect_true(all(is.na(unlist(out[smoother_fields]))))

  out <- call(mod, disturbances = FALSE)
  expect_equal(out[c(filter_fields, "r", "N", smoother_fields)],
               full[c(filter_fields, "r", "N", smoother_fields)])
  expect_true(all(is.na(unlist(out[disturbance_fields]))))

  out <- call(mod, smoother = FALSE, disturbances = FALSE)
  expect_equal(out[filter_fields], full[filter_fields])
  expect_true(all(is.na(unlist(out[c("r", "N", smoother_fields,
                                     disturbance_fields)]))))

  # Non-diagonal H needs the smoothed V for V_eps, even without smoother
  mod$H <- matrix(c(1, 0.6, 0.6, 2), 2)
  full <- run_kfas(mod, y)
  out <- call(mod, smoother = FALSE)
  expect_equal(out[disturbance_fields], full[disturbance_fields])
  expect_true(all(is.na(unlist(out[smoother_fields]))))
})
