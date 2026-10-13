# TODO

Planned features for `kfas()` (`src/kfas0.cpp`, univariate treatment)
and the multivariate routine (`src/kfas.cpp`). Notation follows Durbin &
Koopman (2012):

    y_t       = Z_t alpha_t + eps_t,       eps_t ~ N(0, H_t)
    alpha_t+1 = T_t alpha_t + R_t eta_t,   eta_t ~ N(0, Q_t)

## Features

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
