kfas <- function(
  obs,
  obsmat,
  obsvar,
  statemat,
  vartrans,
  statevar,
  inita,
  initp
) {
  kfas <- .Call(
    "kfas0",
    t(obs),
    obsmat,
    obsvar,
    statemat,
    vartrans,
    statevar,
    inita,
    initp
  )

  result <- list()

  n <- nrow(obs)
  p <- ncol(obs)
  m <- nrow(vartrans)
  q <- ncol(vartrans)

  result$v <- t(kfas[[1]])
  result$F <- array(kfas[[2]], c(p, n))
  result$att <- t(kfas[[3]])
  result$Ptt <-  array(kfas[[4]], c(m, m, n))
  result$a <- t(kfas[[5]])
  result$P <-  array(kfas[[6]], c(m, m, n + 1))
  result$K <-  array(kfas[[7]], c(m, p, n))
  result$r <- t(kfas[[8]])
  result$N <-  array(kfas[[9]], c(m, m, n + 1))
  result$epshat <- t(kfas[[10]])
  result$Veps <- array(kfas[[11]], c(p, n))
  result$etahat <- t(kfas[[12]])
  result$Veta <- array(kfas[[13]], c(q, q, n))
  result$alphahat <- t(kfas[[14]])
  result$V <-  array(kfas[[15]], c(m, m, n))

  return(result)
}
