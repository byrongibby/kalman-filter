# TODO

Planned features for `kfas()` (`src/kfas0.cpp`, univariate treatment)
and the multivariate routine (`src/kfas.cpp`). Notation follows Durbin &
Koopman (2012):

    y_t       = Z_t alpha_t + eps_t,       eps_t ~ N(0, H_t)
    alpha_t+1 = T_t alpha_t + R_t eta_t,   eta_t ~ N(0, Q_t)

## Features

- [ ] **Log-likelihood.** Needed for maximum likelihood estimation of
  model parameters. In the univariate treatment:
  `logLik = -0.5 * sum(log(2 pi) + log(F_ti) + v_ti^2 / F_ti)` over
  observations with `F_ti > 0`, plus `-sum(log(diag(L_t)))` when H is
  non-diagonal and the model is Cholesky-transformed (Jacobian of
  `y* = L^-1 y`).

- [ ] **Missing observations.** `NA` in `y` currently turns every later
  output into `NA`. In the univariate treatment, skip the update for each
  missing `y_ti`: no filter update (`K = 0`, `Finv = 0`), so the smoother
  passes `r` and `N` through unchanged. For non-diagonal H, transform only
  the observed sub-vector of each `y_t`.

- [ ] **Exact diffuse initialisation.** Currently a diffuse initial state
  is approximated by a large `P1`, which loses precision (relative error
  around 1e-3 in `V` at t = 2 with `P1 = 1e7 I`). Implement the exact
  initialisation of Durbin & Koopman (ch. 5, univariate form in
  section 6.4) with `P1 = P1_star + kappa * P1_inf`, as KFAS does with
  `P1inf`.

- [ ] **Mean adjustments.** Allow intercepts in both equations:

      y_t       = d_t + Z_t alpha_t + eps_t
      alpha_t+1 = c_t + T_t alpha_t + R_t eta_t

  with `d_t` (p x 1) and `c_t` (m x 1), each time-invariant or
  time-varying via the same stride mechanism as the system matrices.
  The filter changes are `v_t = y_t - d_t - Z_t a_t` and
  `a_t+1 = c_t + T_t att_t`; the smoother is unchanged. Covers
  exogenous regressors on the observations (`d_t = X_t beta`) without
  adding them to the state.

## Possible extensions

- [ ] Full `Var(eps_t | y) = Z_t V_t Z_t'` (p x p) for non-diagonal H;
  only the diagonal is returned now.
- [ ] Use `dsymm` for the symmetric products `T P T'` and `P N P`.
- [ ] Steady-state early exit for fully time-invariant models without
  missing data: once `P` converges, stop updating `P`, `F` and `K`.
