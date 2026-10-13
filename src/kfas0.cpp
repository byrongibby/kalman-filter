// Kalman Filter (univariate treatment)
// Byron Botha 2025

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "kfas_utils.h"

// True if every one of the nH p x p slices of H is diagonal
static bool is_diagonal(double *H, int p, int nH) {
  for (ptrdiff_t k = 0; k < nH; k++)
    for (int j = 0; j < p; j++)
      for (int i = 0; i < p; i++)
        if (i != j && H[(k * p * p) + (j * p) + i] != 0.0) return false;

  return true;
}

// Lower Cholesky factor L of each H slice, H = L L'. Only the lower
// triangle of L is set; the triangular routines using it ignore the rest.
static void cholesky(double *H, int p, int nH, double *L) {
  const char *lo = "l";
  int info;
  FC_LEN_T len_1 = 1;

  memcpy(L, H, (size_t) p * p * nH * sizeof(double));
  for (ptrdiff_t k = 0; k < nH; k++) {
    F77_CALL(dpotrf)(lo, &p, L + (k * p * p), &p, &info, len_1);
    if (info != 0) {
      error("H is not positive definite at time %lld", (long long) k + 1);
    }
  }
}

static void kfas0_c(
    int n,
    int p,
    int m,
    int q,
    double *y,
    double *Z,
    double *H,
    double *T,
    double *R,
    double *Q,
    int dZ,
    int dH,
    int dT,
    int dR,
    int dQ,
    double *v,
    double *F,
    double *att,
    double *Ptt,
    double *a,
    double *P,
    double *K,
    double *r,
    double *N,
    double *epshat,
    double *V_eps,
    double *etahat,
    double *V_eta,
    double *alphahat,
    double *V,
    double *loglik,
    bool smoother,
    bool disturbances
) {
  // Scalars required for BLAS/LAPACK routines
  const char *tr = "t";
  const char *nt = "n";
  const char *up = "u";
  int inc = 1;
  double pos = 1.0;
  double neg = -1.0;
  double zero = 0.0;

  // String lengths for cstring args
  FC_LEN_T len_1 = 1; 
  FC_LEN_T len_2 = 1;

  // Temporary storage for intermediate results
  // (R_alloc memory is reclaimed by R when the .Call returns)
  double *Finv = (double *) R_alloc((size_t) n * p, sizeof(double));
  double *m1   = (double *) R_alloc(m,     sizeof(double));
  double *mq   = (double *) R_alloc(m * q, sizeof(double));
  double *mm   = (double *) R_alloc(m * m, sizeof(double));
  double *qm_1 = (double *) R_alloc(q * m, sizeof(double));
  double *qm_2 = mq;
  double *RQR  = (double *) R_alloc(m * m, sizeof(double));
  double c;

  // RQR' and QR' are computed once unless R or Q vary over time
  bool varying_RQ = (dR != 0 || dQ != 0);
  if (!varying_RQ) {
    compute_RQR(m, q, R, Q, mq, RQR);
    compute_QR(m, q, R, Q, qm_1);
  }

  // Initialise v = y
  memcpy(v, y, (size_t) n * p * sizeof(double));

  // Log-likelihood (7.2): sum of log(F) + v^2 / F over informative
  // observations, with the constant added at the end
  double ll_sum = 0.0;
  ptrdiff_t n_obs = 0;
  
  // Filter
  for (ptrdiff_t t = 0; t < n; t++) {
    double *Z_t = Z + (t * dZ);
    double *H_t = H + (t * dH);
    double *T_t = T + (t * dT);

    memcpy(att + (t * m), a + (t * m), m * sizeof(double));
    memcpy(Ptt + (t * m * m), P + (t * m * m), m * m * sizeof(double));
    
    for (int i = 0; i < p; i++) {
      // A missing observation carries no information: no update, and
      // Finv = 0 makes the smoother pass r and N through unchanged
      if (ISNAN(y[(t * p) + i])) {
        v[(t * p) + i] = NA_REAL;
        F[(t * p) + i] = NA_REAL;
        Finv[(t * p) + i] = 0.0;
        fill_na(K + (t * m * p) + (i * m), m);
        continue;
      }

      // Innovation (6.13)
      v[(t * p) + i] -= F77_CALL(ddot)(
        &m, Z_t + i, &p, att + (t * m), &inc);
      F[(t * p) + i] = H_t[(i * p) + i];

      // M = Ptt Z_i' (returned as K), F = H_i + Z_i M
      F77_CALL(dsymv)(
        up, &m,
        &pos, Ptt + (t * m * m), &m, Z_t + i, &p,
        &zero, K + (t * m * p) + (i * m), &inc, len_1);
      F[(t * p) + i] += F77_CALL(ddot)(
        &m, K + (t * m * p) + (i * m), &inc, Z_t + i, &p);

      // F is numerically zero when it is within rounding error of the
      // terms summed to form it; the observation then carries no
      // information, so skip the update (Finv = 0 also makes the smoother
      // pass r and N through unchanged)
      double scale = fabs(H_t[(i * p) + i]);
      for (int j = 0; j < m; j++) {
        scale += fabs(Z_t[i + (j * p)] * K[(t * m * p) + (i * m) + j]);
      }
      if (F[(t * p) + i] <= sqrt(DBL_EPSILON) * scale) {
        Finv[(t * p) + i] = 0.0;
        memset(K + (t * m * p) + (i * m), 0, m * sizeof(double));
        continue;
      }

      Finv[(t * p) + i] = 1.0 / F[(t * p) + i];
      ll_sum += log(F[(t * p) + i]) +
        v[(t * p) + i] * v[(t * p) + i] * Finv[(t * p) + i];
      n_obs++;

      // Update state (6.12) with gain M / F: att += M v / F,
      // Ptt -= M M' / F
      c = v[(t * p) + i] * Finv[(t * p) + i];
      F77_CALL(daxpy)(
        &m, &c, K + (t * m * p) + (i * m), &inc,
        att + (t * m), &inc);
      c = -Finv[(t * p) + i];
      F77_CALL(dger)(
        &m, &m,
        &c, K + (t * m * p) + (i * m), &inc, K + (t * m * p) + (i * m), &inc,
        Ptt + (t * m * m), &m);
    }

    // Predict state (6.14)
    F77_CALL(dgemv)(
      nt, &m, &m,
      &pos, T_t, &m, att + (t * m), &inc,
      &zero, a + ((t + 1) * m), &inc, len_1);
    if (varying_RQ) compute_RQR(m, q, R + (t * dR), Q + (t * dQ), mq, RQR);
    memcpy(P + ((t + 1) * m * m), RQR, m * m * sizeof(double));
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, T_t, &m, Ptt + (t * m * m), &m,
      &zero, mm, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &m, &m, &m,
      &pos, mm, &m, T_t, &m,
      &pos, P + ((t + 1) * m * m), &m, len_1, len_2);
  }

  *loglik = -0.5 * ((double) n_obs * LOG_2PI + ll_sum);

  if (smoother || disturbances) {
    if (smoother) {
      // Initialise alphahat = a, and V = P
      memcpy(alphahat, a, (size_t) n * m * sizeof(double));
      memcpy(V, P, (size_t) n * m * m * sizeof(double));
    }

    // Smooth
    for (ptrdiff_t t = n - 1; t >= 0; t--) {
      double *Z_t = Z + (t * dZ);
      double *H_t = H + (t * dH);
      double *T_t = T + (t * dT);
      double *Q_t = Q + (t * dQ);

      // Previous weighted sum of innovations (6.15.2)
      F77_CALL(dgemv)(
          tr, &m, &m,
          &pos, T_t, &m, r + ((t + 1) * m), &inc,
          &zero, r + (t * m), &inc, len_1);
      F77_CALL(dgemm)(
          tr, nt, &m, &m, &m,
          &pos, T_t, &m, N + ((t + 1) * m * m), &m,
          &zero, mm, &m, len_1, len_2);
      F77_CALL(dgemm)(
          nt, nt, &m, &m, &m,
          &pos, mm, &m, T_t, &m,
          &zero, N + (t * m * m), &m, len_1, len_2);

      if (disturbances) {
        // Smoothed eta disturbances (4.69)
        if (varying_RQ) compute_QR(m, q, R + (t * dR), Q_t, qm_1);
        memcpy(V_eta + (t * q * q), Q_t, q * q * sizeof(double));
        F77_CALL(dgemv)(
          nt, &q, &m,
          &pos, qm_1, &q, r + ((t + 1) * m), &inc,
          &zero, etahat + (t * q), &inc, len_1);
        F77_CALL(dgemm)(
          nt, nt, &q, &m, &m,
          &pos, qm_1, &q, N + ((t + 1) * m * m), &m,
          &zero, qm_2, &q, len_1, len_2);
        F77_CALL(dgemm)(
          nt, tr, &q, &q, &m,
          &neg, qm_2, &q, qm_1, &q,
          &pos, V_eta + (t * q * q), &q, len_1, len_2);
      }

      for (int i = p - 1; i >= 0; i--) {
        double Finv_ti = Finv[(t * p) + i];

        // Skipped (missing or uninformative) observations leave r and N
        // unchanged; epsilon keeps its prior mean and variance
        if (Finv_ti == 0.0) {
          if (disturbances) {
            epshat[(t * p) + i] = 0.0;
            V_eps[(t * p) + i] = H_t[(i * p) + i];
          }
          continue;
        }

        // L = I - K z' is rank-1 (z = Z_i', K = M / F), so it is never
        // formed:
        //   L'r + F^-1 v z    = r + (F^-1 v - K'r) z
        //   L'NL + F^-1 z z'  = N - z w' - w z' + (s + F^-1) z z'
        // with w = N K and s = K'w
        double *M_ti = K + (t * m * p) + (i * m);
        double Kr = Finv_ti * F77_CALL(ddot)(&m, M_ti, &inc, r + (t * m), &inc);
        F77_CALL(dsymv)(
          up, &m,
          &Finv_ti, N + (t * m * m), &m, M_ti, &inc,
          &zero, m1, &inc, len_1);
        double s = Finv_ti * F77_CALL(ddot)(&m, M_ti, &inc, m1, &inc);
        double Finv_v = Finv_ti * v[(t * p) + i];

        if (disturbances) {
          // Smoothed epsilon disturbances
          epshat[(t * p) + i] =
            H_t[(i * p) + i] * (Finv_v - Kr);
          V_eps[(t * p) + i] = H_t[(i * p) + i] -
            H_t[(i * p) + i] * H_t[(i * p) + i] * (Finv_ti + s);
        }

        // Weighted sum of innovations (6.15.1)
        c = Finv_v - Kr;
        F77_CALL(daxpy)(&m, &c, Z_t + i, &p, r + (t * m), &inc);
        F77_CALL(dger)(
          &m, &m,
          &neg, Z_t + i, &p, m1, &inc,
          N + (t * m * m), &m);
        F77_CALL(dger)(
          &m, &m,
          &neg, m1, &inc, Z_t + i, &p,
          N + (t * m * m), &m);
        c = s + Finv_ti;
        F77_CALL(dger)(
          &m, &m,
          &c, Z_t + i, &p, Z_t + i, &p,
          N + (t * m * m), &m);
      }

      if (smoother) {
        // Smoothed state (4.44)
        F77_CALL(dsymv)(
          up, &m,
          &pos, P + (t * m * m), &m, r + (t * m), &inc,
          &pos, alphahat + (t * m), &inc, len_1);
        F77_CALL(dgemm)(
          nt, nt, &m, &m, &m,
          &pos, P + (t * m * m), &m, N + (t * m * m), &m,
          &zero, mm, &m, len_1, len_2);
        F77_CALL(dgemm)(
          nt, nt, &m, &m, &m,
          &neg, mm, &m, P + (t * m * m), &m,
          &pos, V + (t * m * m), &m, len_1, len_2);
      }
    }
  }
}

extern "C" SEXP kfas0(
    SEXP y,
    SEXP Z,
    SEXP H,
    SEXP T,
    SEXP R,
    SEXP Q,
    SEXP a1,
    SEXP P1,
    SEXP smoother,
    SEXP disturbances
) {
  check_matrix(y, "y");
  check_matrix(R, "R");
  check_double(Z, -1, "Z");
  check_double(H, -1, "H");
  check_double(T, -1, "T");
  check_double(Q, -1, "Q");

  SEXP dim_y = PROTECT(getAttrib(y, R_DimSymbol));
  SEXP dim_R = PROTECT(getAttrib(R, R_DimSymbol));
  
  int p = INTEGER(dim_y)[0];
  int n = INTEGER(dim_y)[1];
  int m = INTEGER(dim_R)[0];
  int q = INTEGER(dim_R)[1];

  int dZ = stride(Z, p * m, n, "Z");
  int dH = stride(H, p * p, n, "H");
  int dT = stride(T, m * m, n, "T");
  int dR = stride(R, m * q, n, "R");
  int dQ = stride(Q, q * q, n, "Q");
  check_double(a1, m, "a1");
  check_double(P1, (R_xlen_t) m * m, "P1");

  bool do_smoother = asLogical(smoother) == TRUE;
  bool do_disturbances = asLogical(disturbances) == TRUE;

  SEXP v        = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP F        = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP att      = PROTECT(allocMatrix(REALSXP, m    , n));
  SEXP Ptt      = PROTECT(alloc3DArray(REALSXP, m, m, n));
  SEXP a        = PROTECT(allocMatrix(REALSXP, m    , n + 1));
  SEXP P        = PROTECT(alloc3DArray(REALSXP, m, m, n + 1));
  SEXP K        = PROTECT(alloc3DArray(REALSXP, m, p, n));
  SEXP r        = PROTECT(allocMatrix(REALSXP, m    , n + 1));
  SEXP N        = PROTECT(alloc3DArray(REALSXP, m, m, n + 1));
  SEXP epshat   = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP V_eps    = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP etahat   = PROTECT(allocMatrix(REALSXP, q    , n));
  SEXP V_eta    = PROTECT(alloc3DArray(REALSXP, q, q, n));
  SEXP alphahat = PROTECT(allocMatrix(REALSXP, m    , n));
  SEXP V        = PROTECT(alloc3DArray(REALSXP, m, m, n));
  SEXP loglik   = PROTECT(allocVector(REALSXP, 1));

  int list_length = 16;

  SEXP result = PROTECT(allocVector(VECSXP, list_length));

  double *y_ptr        = REAL(y);
  double *Z_ptr        = REAL(Z);
  double *H_ptr        = REAL(H);
  double *T_ptr        = REAL(T);
  double *R_ptr        = REAL(R);
  double *Q_ptr        = REAL(Q);
  double *a1_ptr       = REAL(a1);
  double *P1_ptr       = REAL(P1);
  double *v_ptr        = REAL(v);
  double *F_ptr        = REAL(F);
  double *att_ptr      = REAL(att);
  double *Ptt_ptr      = REAL(Ptt);
  double *a_ptr        = REAL(a);
  double *P_ptr        = REAL(P);
  double *K_ptr        = REAL(K);
  double *r_ptr        = REAL(r);
  double *N_ptr        = REAL(N);
  double *epshat_ptr   = REAL(epshat);
  double *V_eps_ptr    = REAL(V_eps);
  double *etahat_ptr   = REAL(etahat);
  double *V_eta_ptr    = REAL(V_eta);
  double *alphahat_ptr = REAL(alphahat);
  double *V_ptr        = REAL(V);

  memcpy(a_ptr, a1_ptr, m * sizeof(double));
  memcpy(P_ptr, P1_ptr, m * m * sizeof(double));

  memset(r_ptr + ((ptrdiff_t) n * m), 0, m * sizeof(double));
  memset(N_ptr + ((ptrdiff_t) n * m * m), 0, m * m * sizeof(double));

  // Scalars required for BLAS routines
  const char *lf = "l";
  const char *lo = "l";
  const char *nt = "n";
  const char *nu = "n";
  const char *up = "u";
  int inc = 1;
  double pos = 1.0;
  double zero = 0.0;
  FC_LEN_T len_1 = 1;

  // The univariate treatment needs diagonal H. Otherwise factor H = L L'
  // and filter the transformed model y* = L^-1 y, Z* = L^-1 Z, H* = I
  double *y_in = y_ptr;
  double *Z_in = Z_ptr;
  double *H_in = H_ptr;
  int dZ_in = dZ;
  int dH_in = dH;
  int dL = dH;
  double *L = NULL;
  bool diagonal = is_diagonal(H_ptr, p, dH ? n : 1);

  // Observed elements of each y_t: count n_obs[t] and indices obs[t * p + j]
  bool has_missing = false;
  int *n_obs = NULL;
  int *obs = NULL;
  if (!diagonal) {
    n_obs = (int *) R_alloc(n, sizeof(int));
    obs = (int *) R_alloc((size_t) n * p, sizeof(int));
    for (ptrdiff_t t = 0; t < n; t++) {
      n_obs[t] = 0;
      for (int i = 0; i < p; i++) {
        if (!ISNAN(y_ptr[(t * p) + i])) obs[(t * p) + n_obs[t]++] = i;
      }
      if (n_obs[t] < p) has_missing = true;
    }
  }

  if (!diagonal && !has_missing) {
    L = (double *) R_alloc((size_t) p * p * (dH ? n : 1), sizeof(double));
    cholesky(H_ptr, p, dH ? n : 1, L);

    y_in = (double *) R_alloc((size_t) n * p, sizeof(double));
    memcpy(y_in, y_ptr, (size_t) n * p * sizeof(double));
    for (ptrdiff_t t = 0; t < n; t++) {
      F77_CALL(dtrsv)(
        lo, nt, nu, &p,
        L + (t * dL), &p, y_in + (t * p), &inc, len_1, len_1, len_1);
    }

    // Z* varies over time if either Z or H does
    dZ_in = (dZ != 0 || dH != 0) ? p * m : 0;
    int nZ = dZ_in ? n : 1;
    Z_in = (double *) R_alloc((size_t) p * m * nZ, sizeof(double));
    for (ptrdiff_t k = 0; k < nZ; k++) {
      memcpy(Z_in + (k * p * m), Z_ptr + (k * dZ), p * m * sizeof(double));
      F77_CALL(dtrsm)(
        lf, lo, nt, nu, &p, &m,
        &pos, L + (k * dL), &p, Z_in + (k * p * m), &p,
        len_1, len_1, len_1, len_1);
    }
  } else if (!diagonal) {
    // With missing observations, transform each y_t using only its observed
    // elements: H_oo = L_t L_t' (stored compactly, leading dimension
    // n_obs[t]). Missing elements of y* stay NA and are skipped.
    dL = p * p;
    L = (double *) R_alloc((size_t) n * p * p, sizeof(double));
    y_in = (double *) R_alloc((size_t) n * p, sizeof(double));
    fill_na(y_in, (R_xlen_t) n * p);
    dZ_in = p * m;
    Z_in = (double *) S_alloc((size_t) n * p * m, sizeof(double));
    double *yc = (double *) R_alloc(p, sizeof(double));
    double *Zc = (double *) R_alloc(p * m, sizeof(double));
    int info;

    for (ptrdiff_t t = 0; t < n; t++) {
      int po = n_obs[t];
      int *idx = obs + (t * p);
      double *H_t = H_ptr + (t * dH);
      double *Z_t = Z_ptr + (t * dZ);
      double *L_t = L + (t * dL);
      if (po == 0) continue;

      for (int k = 0; k < po; k++) {
        for (int j = 0; j < po; j++) L_t[j + (k * po)] = H_t[idx[j] + (idx[k] * p)];
      }
      F77_CALL(dpotrf)(lo, &po, L_t, &po, &info, len_1);
      if (info != 0) {
        error("H is not positive definite at time %lld", (long long) t + 1);
      }

      for (int j = 0; j < po; j++) yc[j] = y_ptr[(t * p) + idx[j]];
      F77_CALL(dtrsv)(
        lo, nt, nu, &po,
        L_t, &po, yc, &inc, len_1, len_1, len_1);
      for (int j = 0; j < po; j++) y_in[(t * p) + idx[j]] = yc[j];

      for (int c = 0; c < m; c++) {
        for (int j = 0; j < po; j++) Zc[j + (c * po)] = Z_t[idx[j] + (c * p)];
      }
      F77_CALL(dtrsm)(
        lf, lo, nt, nu, &po, &m,
        &pos, L_t, &po, Zc, &po,
        len_1, len_1, len_1, len_1);
      for (int c = 0; c < m; c++) {
        for (int j = 0; j < po; j++) {
          Z_in[(t * p * m) + idx[j] + (c * p)] = Zc[j + (c * po)];
        }
      }
    }
  }

  if (!diagonal) {
    H_in = (double *) S_alloc(p * p, sizeof(double));
    for (int i = 0; i < p; i++) H_in[(i * p) + i] = 1.0;
    dH_in = 0;
  }

  // Original-scale V_eps for non-diagonal H needs the smoothed V
  bool run_smoother = do_smoother || (do_disturbances && !diagonal);

  kfas0_c(
    n,
    p,
    m,
    q,
    y_in,
    Z_in,
    H_in,
    T_ptr,
    R_ptr,
    Q_ptr,
    dZ_in,
    dH_in,
    dT,
    dR,
    dQ,
    v_ptr,
    F_ptr,
    att_ptr,
    Ptt_ptr,
    a_ptr,
    P_ptr,
    K_ptr,
    r_ptr,
    N_ptr,
    epshat_ptr,
    V_eps_ptr,
    etahat_ptr,
    V_eta_ptr,
    alphahat_ptr,
    V_ptr,
    REAL(loglik),
    run_smoother,
    do_disturbances
  );

  // L_t has leading dimension n_obs[t] (p when nothing is missing)
  if (!diagonal) {
    // Jacobian of y* = L^-1 y: log f(y) = log f(y*) - sum_t log det L_t
    double log_det = 0.0;
    for (ptrdiff_t t = 0; t < n; t++) {
      for (int j = 0; j < n_obs[t]; j++) {
        log_det += log(L[(t * dL) + (j * n_obs[t]) + j]);
      }
    }
    REAL(loglik)[0] -= log_det;
  }

  if (!diagonal && do_disturbances) {
    // Return the epsilon disturbances in the original scale. Observed
    // elements: epshat_o = L epshat*_o (= y_o - Z_o alphahat) and
    // Var(eps_i | y) = Z_i V Z_i'. Missing elements, from the conditional
    // distribution given eps_o with g = H_oo^-1 H_oi:
    //   epshat_i = g' epshat_o
    //   Var(eps_i | y) = H_ii - H_io g + (Z_o' g)' V (Z_o' g)
    // (diagonal only). v, F and K stay in the transformed scale.
    double *m1 = (double *) R_alloc(m, sizeof(double));
    double *u  = (double *) R_alloc(m, sizeof(double));
    double *e  = (double *) R_alloc(p, sizeof(double));
    double *h  = (double *) R_alloc(p, sizeof(double));
    double *g  = (double *) R_alloc(p, sizeof(double));
    int info;

    for (ptrdiff_t t = 0; t < n; t++) {
      int po = n_obs[t];
      int *idx = obs + (t * p);
      double *H_t = H_ptr + (t * dH);
      double *Z_t = Z_ptr + (t * dZ);
      double *L_t = L + (t * dL);
      double *V_t = V_ptr + (t * m * m);

      for (int j = 0; j < po; j++) e[j] = epshat_ptr[(t * p) + idx[j]];
      if (po > 0) {
        F77_CALL(dtrmv)(
          lo, nt, nu, &po,
          L_t, &po, e, &inc, len_1, len_1, len_1);
      }

      int j = 0;
      for (int i = 0; i < p; i++) {
        if (j < po && idx[j] == i) {
          // Observed
          epshat_ptr[(t * p) + i] = e[j++];
          F77_CALL(dsymv)(
            up, &m,
            &pos, V_t, &m, Z_t + i, &p,
            &zero, m1, &inc, len_1);
          V_eps_ptr[(t * p) + i] = F77_CALL(ddot)(&m, Z_t + i, &p, m1, &inc);
        } else if (po == 0) {
          // Missing, with nothing observed at time t
          epshat_ptr[(t * p) + i] = 0.0;
          V_eps_ptr[(t * p) + i] = H_t[(i * p) + i];
        } else {
          // Missing
          for (int k = 0; k < po; k++) h[k] = H_t[idx[k] + (i * p)];
          memcpy(g, h, po * sizeof(double));
          F77_CALL(dpotrs)(lo, &po, &inc, L_t, &po, g, &po, &info, len_1);
          memset(u, 0, m * sizeof(double));
          for (int k = 0; k < po; k++) {
            F77_CALL(daxpy)(&m, g + k, Z_t + idx[k], &p, u, &inc);
          }
          F77_CALL(dsymv)(
            up, &m,
            &pos, V_t, &m, u, &inc,
            &zero, m1, &inc, len_1);
          epshat_ptr[(t * p) + i] = F77_CALL(ddot)(&po, g, &inc, e, &inc);
          V_eps_ptr[(t * p) + i] = H_t[(i * p) + i] -
            F77_CALL(ddot)(&po, h, &inc, g, &inc) +
            F77_CALL(ddot)(&m, u, &inc, m1, &inc);
        }
      }
    }
  }

  // Outputs that were not requested are NA rather than uninitialised
  if (!do_smoother) {
    fill_na(alphahat_ptr, XLENGTH(alphahat));
    fill_na(V_ptr, XLENGTH(V));
  }
  if (!do_disturbances) {
    fill_na(epshat_ptr, XLENGTH(epshat));
    fill_na(V_eps_ptr, XLENGTH(V_eps));
    fill_na(etahat_ptr, XLENGTH(etahat));
    fill_na(V_eta_ptr, XLENGTH(V_eta));
  }
  if (!do_smoother && !do_disturbances) {
    fill_na(r_ptr, XLENGTH(r));
    fill_na(N_ptr, XLENGTH(N));
  }

  SET_VECTOR_ELT(result,  0, v);
  SET_VECTOR_ELT(result,  1, F);
  SET_VECTOR_ELT(result,  2, att);
  SET_VECTOR_ELT(result,  3, Ptt);
  SET_VECTOR_ELT(result,  4, a);
  SET_VECTOR_ELT(result,  5, P);
  SET_VECTOR_ELT(result,  6, K);
  SET_VECTOR_ELT(result,  7, r);
  SET_VECTOR_ELT(result,  8, N);
  SET_VECTOR_ELT(result,  9, epshat);
  SET_VECTOR_ELT(result, 10, V_eps);
  SET_VECTOR_ELT(result, 11, etahat);
  SET_VECTOR_ELT(result, 12, V_eta);
  SET_VECTOR_ELT(result, 13, alphahat);
  SET_VECTOR_ELT(result, 14, V);
  SET_VECTOR_ELT(result, 15, loglik);

  UNPROTECT(19);

  return result;
}
