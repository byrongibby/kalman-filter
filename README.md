# kfilter

Kalman filter and smoother for linear Gaussian state space models, in
C-style C++ on top of BLAS and LAPACK.

    y_t       = d_t + Z_t alpha_t + eps_t,       eps_t ~ N(0, H_t)
    alpha_t+1 = c_t + T_t alpha_t + R_t eta_t,   eta_t ~ N(0, Q_t)

`kfas()` runs the filter, the state smoother and the disturbance smoother
using the univariate treatment of multivariate series (Durbin & Koopman,
2012, section 6.4). It supports:

- time-varying system matrices: any of Z, H, T, R and Q can be an array
  with one slice per time step;
- intercepts `d_t` and `c_t` in the observation and state equations,
  time-invariant or time-varying;
- non-diagonal H, through a Cholesky transformation of the model;
- missing observations (`NA` in the data), in any pattern;
- exact diffuse initialisation for states with unknown starting values;
- the Gaussian log-likelihood, for maximum likelihood estimation;
- filtering only, or filtering and state smoothing only, via the
  `smoother` and `disturbances` arguments.

`kfas_loglik()` computes only the log-likelihood, without storing the
filter's per-step output, for use inside optimisers.

Two functions draw the states from their joint posterior distribution:

- `simulation_smoother()` uses the simulation smoother of Durbin & Koopman
  (2002) and supports every model `kfas()` does;
- `precision_sampler()` uses the precision-based sampler of Chan &
  Jeliazkov (2009), which factors the banded posterior precision of all
  the states at once. It is faster, but needs non-singular state
  disturbance and observation variances.

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

Durbin, J. and Koopman, S. J. (2002). A simple and efficient simulation
smoother for state space time series analysis. *Biometrika*, 89(3),
603-615.

Chan, J. C. C. and Jeliazkov, I. (2009). Efficient simulation and
integrated likelihood estimation in state space models. *International
Journal of Mathematical Modelling and Numerical Optimisation*, 1(1/2),
101-120.
