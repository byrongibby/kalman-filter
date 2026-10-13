# src/kfas.cpp (multivariate treatment) is not exported, but is tested so it
# can't drift from the univariate kfas0 routine behind kfas()

models <- list(
  local_level = local_level(),
  local_trend = local_trend(),
  bivariate   = bivariate()
)

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

    # Filtered and smoothed states are invariant to the treatment;
    # the intermediate a/P within a time step are not compared
    for (field in c("a", "P", "att", "Ptt", "alphahat", "V")) {
      expect_equal(multi[[field]], uni[[field]], tolerance = 1e-8,
                   label = paste(name, field))
    }
  }
})

test_that("multivariate routine matches KFAS::KFS", {
  skip_if_not_installed("KFAS")

  for (name in names(models)) {
    mod <- models[[name]]
    y <- simulate_y(mod)
    multi <- run_kfas_multivariate(mod, y)
    ref <- run_reference(mod, y)

    for (field in c("a", "P", "att", "Ptt", "alphahat", "V")) {
      expect_same_values(multi[[field]], ref[[field]])
    }
  }
})
