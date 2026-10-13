test_that("mvrnorm returns an n x d matrix of finite draws", {
  sigma <- matrix(c(2, 0.5, 0.5, 1), 2)
  x <- mvrnorm(100L, c(1, -1), sigma)

  expect_equal(dim(x), c(100, 2))
  expect_true(all(is.finite(x)))
})

test_that("mvrnorm handles a single univariate draw", {
  x <- mvrnorm(1L, 0, matrix(1))

  expect_equal(dim(x), c(1, 1))
  expect_true(is.finite(x))
})

test_that("mvrnorm accepts a double for n", {
  expect_equal(dim(mvrnorm(10, 0, matrix(1))), c(10, 1))
})

test_that("mvrnorm draws have the requested mean and covariance", {
  mu <- c(1, -1, 0.5)
  sigma <- matrix(c(2,   0.5, 0.3,
                    0.5, 1,   -0.2,
                    0.3, -0.2, 0.5), 3)
  n <- 20000L
  x <- mvrnorm(n, mu, sigma)

  # Sample mean within 5 standard errors of mu
  expect_true(all(abs(colMeans(x) - mu) < 5 * sqrt(diag(sigma) / n)))
  expect_equal(cov(x), sigma, tolerance = 0.05)
})

test_that("consecutive mvrnorm calls give different draws", {
  expect_false(identical(mvrnorm(10L, 0, matrix(1)),
                         mvrnorm(10L, 0, matrix(1))))
})

test_that("every position in a draw is random", {
  # Each of the 5 positions should vary across calls, including the last
  draws <- replicate(200, mvrnorm(5L, 0, matrix(1))[, 1])
  expect_true(all(apply(draws, 1, sd) > 0.5))
})

test_that("mvrnorm accepts integer mu and sigma", {
  set.seed(5)
  x <- mvrnorm(10L, c(1L, 2L), diag(2L))
  set.seed(5)
  expect_equal(x, mvrnorm(10L, c(1, 2), diag(2)))
})

test_that("mvrnorm rejects a covariance that is not positive definite", {
  expect_error(mvrnorm(5L, c(0, 0), matrix(c(1, 2, 2, 1), 2)),
               "sigma is not positive definite")
})

test_that("mvrnorm rejects malformed inputs", {
  sigma <- matrix(c(2, 0.5, 0.5, 1), 2)

  expect_error(mvrnorm(5L, c(0, 0), matrix(c(2, 0.5, 0.4, 1), 2)),
               "sigma must be symmetric")
  expect_error(mvrnorm(5L, 0, 1), "sigma must be a matrix or array")
  expect_error(mvrnorm(5L, c(0, 0), matrix(1, 2, 3)),
               "sigma must be a square matrix")
  expect_error(mvrnorm(5L, 0, sigma), "mu must have 2 elements, not 1")
  expect_error(mvrnorm(5L, c("a", "b"), sigma), "must be numeric")
  for (n in list(0L, -1L, NA)) {
    expect_error(mvrnorm(n, c(0, 0), sigma), "n must be a positive integer")
  }
})
