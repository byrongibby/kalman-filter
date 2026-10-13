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
