kfas <- function(
  obs,
  obsmat,
  obsvar,
  statemat,
  vartrans,
  statevar,
  inita,
  initp,
  smoother = TRUE,
  disturbances = TRUE
) {
  if (!is.matrix(obs)) stop("obs must be an n x p matrix", call. = FALSE)

  x <- .Call(
    "kfas0",
    t(as_double(obs)),
    as_double(obsmat),
    as_double(obsvar),
    as_double(statemat),
    as_double(vartrans),
    as_double(statevar),
    as_double(inita),
    as_double(initp),
    as.logical(smoother),
    as.logical(disturbances)
  )

  result <- list()

  n <- nrow(obs)
  p <- ncol(obs)
  m <- nrow(vartrans)
  q <- ncol(vartrans)

  F <- as.vector(t(matrix(x[[2]], p * n, m)))

  result$v <- t(x[[1]])
  result$F <- array(x[[2]], c(p, n))
  result$att <- t(x[[3]])
  result$Ptt <-  array(x[[4]], c(m, m, n))
  result$a <- t(x[[5]])
  result$P <-  array(x[[6]], c(m, m, n + 1))
  result$K <-  array(x[[7]] * F, c(m, p, n))
  result$r <- t(x[[8]])
  result$N <-  array(x[[9]], c(m, m, n + 1))
  result$epshat <- t(x[[10]])
  result$V_eps <- array(x[[11]], c(p, n))
  result$etahat <- t(x[[12]])
  result$V_eta <- array(x[[13]], c(q, q, n))
  result$alphahat <- t(x[[14]])
  result$V <-  array(x[[15]], c(m, m, n))

  return(result)
}

# Convert to double storage, keeping dimensions; the C code reads doubles
as_double <- function(x) {
  if (!is.numeric(x)) {
    stop(deparse(substitute(x)), " must be numeric", call. = FALSE)
  }
  storage.mode(x) <- "double"
  x
}

mvrnorm <- function(n, mu, sigma) {
  z <- .Call("mvrnorm", n, mu, sigma)

  return(t(z))
}
