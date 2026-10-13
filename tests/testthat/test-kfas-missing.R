all_matrices <- c("Z", "H", "T", "R", "Q")
models <- list(
  local_level  = local_level(),
  local_trend  = local_trend(),
  bivariate    = bivariate(),
  time_varying = time_varying(bivariate(), all_matrices),
  full_H       = full_H(bivariate()),
  full_H_tv    = time_varying(full_H(bivariate()), all_matrices)
)

# About a fifth of the elements missing at random, plus a time step with
# every element missing and, for p > 1, a stretch where one series is
# missing throughout
with_missing <- function(y, seed = 2) {
  set.seed(seed)
  y[sample(length(y), round(length(y) / 5))] <- NA
  y[10, ] <- NA
  if (ncol(y) > 1) y[20:25, 2] <- NA
  y
}

smoothed_fields <- c("alphahat", "V", "epshat", "V_eps", "etahat", "V_eta",
                     "logLik")

test_that("smoothed outputs with missing data equal the exact posterior", {
  for (name in names(models)) {
    mod <- models[[name]]
    y <- with_missing(simulate_y(mod))
    expected <- exact_posterior(mod, y)
    uni <- run_kfas(mod, y)
    multi <- run_kfas_multivariate(mod, y)

    # The reference inverts the full covariance of the observed y, which
    # is ill-conditioned for the trend model (variances grow like t^3), so
    # it is only accurate to about 1e-7; the filter agrees with KFAS to
    # around 1e-13 (see the KFAS test below)
    for (field in smoothed_fields) {
      expect_equal(uni[[field]], expected[[field]], tolerance = 1e-6,
                   label = paste(name, "univariate", field))
      expect_equal(multi[[field]], expected[[field]], tolerance = 1e-6,
                   label = paste(name, "multivariate", field))
    }
  }
})

test_that("kfas with missing data matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  for (name in names(models)) {
    mod <- models[[name]]
    y <- with_missing(simulate_y(mod))
    ref <- run_reference(mod, y)
    uni <- run_kfas(mod, y)
    multi <- run_kfas_multivariate(mod, y)

    fields <- c("att", "Ptt", "a", "P", "alphahat", "V", "etahat", "V_eta")
    # v and F are on a different scale from KFAS's for non-diagonal H
    if (!startsWith(name, "full_H")) fields <- c(fields, "v", "F", "epshat",
                                                 "V_eps")
    for (field in fields) {
      expect_same_values(uni[[field]], ref[[field]])
    }
    for (field in setdiff(fields, c("v", "F"))) {
      expect_same_values(multi[[field]], ref[[field]])
    }
    expect_equal(uni$logLik, ref$logLik, tolerance = 1e-10)
    expect_equal(multi$logLik, ref$logLik, tolerance = 1e-10)
  }
})

test_that("v, F and K are NA exactly at the missing observations", {
  mod <- bivariate()
  y <- with_missing(simulate_y(mod))
  missing <- is.na(y)
  uni <- run_kfas(mod, y)
  multi <- run_kfas_multivariate(mod, y)

  expect_identical(is.na(uni$v), missing)
  expect_identical(is.na(t(uni$F)), missing)
  expect_identical(is.na(multi$v), missing)
  for (t in seq_len(nrow(y))) {
    expect_identical(is.na(uni$K[1, , t]), missing[t, ])
    expect_identical(is.na(multi$K[1, , t]), missing[t, ])
    expect_identical(is.na(diag(as.matrix(multi$F[, , t]))), missing[t, ])
  }
})

test_that("with every observation missing, the smoother returns the prior", {
  mod <- local_trend()
  y <- simulate_y(mod)
  y[] <- NA
  out <- run_kfas(mod, y)

  expect_equal(out$alphahat, out$a[seq_len(nrow(y)), ])
  expect_equal(out$V, out$P[, , seq_len(nrow(y))])
  expect_equal(out$logLik, 0)
  expect_equal(run_kfas_multivariate(mod, y)$alphahat, out$alphahat)
})
