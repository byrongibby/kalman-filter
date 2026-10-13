# kfilter

Kalman filter and smoother for linear Gaussian state space models, in
C-style C++ on top of BLAS and LAPACK.

    y_t       = Z_t alpha_t + eps_t,       eps_t ~ N(0, H_t)
    alpha_t+1 = T_t alpha_t + R_t eta_t,   eta_t ~ N(0, Q_t)

`kfas()` runs the filter, the state smoother and the disturbance smoother
using the univariate treatment of multivariate series (Durbin & Koopman,
2012, section 6.4). It supports:

- time-varying system matrices: any of Z, H, T, R and Q can be an array
  with one slice per time step;
- non-diagonal H, through a Cholesky transformation of the model;
- missing observations (`NA` in the data), in any pattern;
- exact diffuse initialisation for states with unknown starting values;
- the Gaussian log-likelihood, for maximum likelihood estimation;
- filtering only, or filtering and state smoothing only, via the
  `smoother` and `disturbances` arguments.

`kfas_loglik()` computes only the log-likelihood, without storing the
filter's per-step output, for use inside optimisers.

`mvrnorm()` draws from a multivariate normal distribution.

See `?kfas` for the model, argument shapes and outputs, and `TODO.md` for
planned features.

## Example

```r
library(kfilter)

# Local level model for the Nile data (Durbin & Koopman, section 2.10),
# with a diffuse initial level
y <- matrix(as.numeric(Nile))
fit <- kfas(y, obsmat = matrix(1), obsvar = matrix(15099),
            statemat = matrix(1), vartrans = matrix(1),
            statevar = matrix(1469.1), inita = 0, initp = matrix(0),
            initpinf = matrix(1))

plot(as.numeric(Nile), ylab = "Flow")
lines(fit$alphahat[, 1])
```

## Installation and tests

```sh
R CMD INSTALL .
Rscript -e 'testthat::test_local()'
```

The tests compare results against the KFAS package where it is installed.

## Reference

Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space
Methods*, 2nd edition. Oxford University Press.
