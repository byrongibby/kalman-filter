// Kalman Filter (univariate treatment)
// Byron Botha 2025

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

static int kfas0_c(
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
    bool filter,
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
  double *Finv = (double *) R_alloc(n * p, sizeof(double));
  double *m1   = (double *) R_alloc(m,     sizeof(double));
  double *mq   = (double *) R_alloc(m * q, sizeof(double));
  double *mm   = (double *) R_alloc(m * m, sizeof(double));
  double *qm_1 = (double *) R_alloc(q * m, sizeof(double));
  double *qm_2 = mq;
  double *RQR  = (double *) R_alloc(m * m, sizeof(double));
  double c;

  // Time-invariant state disturbance variance RQR' and QR'
  F77_CALL(dgemm)(
    nt, nt, &m, &q, &q,
    &pos, R, &m, Q, &q,
    &zero, mq, &m, len_1, len_2);
  F77_CALL(dgemm)(
    nt, tr, &m, &m, &q,
    &pos, mq, &m, R, &m,
    &zero, RQR, &m, len_1, len_2);
  F77_CALL(dgemm)(
    nt, tr, &q, &m, &q,
    &pos, Q, &q, R, &m,
    &zero, qm_1, &q, len_1, len_2);

  if (filter) {
    // Initialise v = y
    memcpy(v, y, n * p * sizeof(double));
    
    // Filter
    for (int t = 0; t < n; t++) {
      memcpy(att + (t * m), a + (t * m), m * sizeof(double));
      memcpy(Ptt + (t * m * m), P + (t * m * m), m * m * sizeof(double));
      
      for (int i = 0; i < p; i++) {
        // Innovation (6.13)
        memcpy(F + (t * p) + i, H + (i * p) + i, sizeof(double));
        F77_CALL(dgemv)(
          nt, &inc, &m,
          &neg, Z + i, &p, att + (t * m), &inc,
          &pos, v + (t * p) + i, &inc, len_1);
        // M = Ptt Z_i' (stored in K), F = H_i + Z_i M
        F77_CALL(dsymv)(
          up, &m,
          &pos, Ptt + (t * m * m), &m, Z + i, &p,
          &zero, K + (t * m * p) + (i * m), &inc, len_1);
        F[(t * p) + i] += F77_CALL(ddot)(
          &m, K + (t * m * p) + (i * m), &inc, Z + i, &p);

        // K matrix (6.13), K = M / F
        Finv[(t * p) + i] = 1.0 / F[(t * p) + i];
        F77_CALL(dscal)(
          &m, Finv + (t * p) + i, K + (t * m * p) + (i * m), &inc);

        // Update state (6.12)
        F77_CALL(dgemv)(
          nt, &m, &inc,
          &pos, K + (t * m * p) + (i * m), &m, v + (t * p) + i, &inc,
          &pos, att + (t * m), &inc, len_1);
        c = neg * F[(t * p) + i];
        F77_CALL(dger)(
          &m, &m,
          &c, K + (t * m * p) + (i * m), &inc, K + (t * m * p) + (i * m), &inc,
          Ptt + (t * m * m), &m);
      }

      // Predict state (6.14)
      F77_CALL(dgemv)(
        nt, &m, &m,
        &pos, T, &m, att + (t * m), &inc,
        &zero, a + ((t + 1) * m), &inc, len_1);
      memcpy(P + ((t + 1) * m * m), RQR, m * m * sizeof(double));
      F77_CALL(dgemm)(
        nt, nt, &m, &m, &m,
        &pos, T, &m, Ptt + (t * m * m), &m,
        &zero, mm, &m, len_1, len_2);
      F77_CALL(dgemm)(
        nt, tr, &m, &m, &m,
        &pos, mm, &m, T, &m,
        &pos, P + ((t + 1) * m * m), &m, len_1, len_2);
    }
  }

  if (smoother || disturbances) {
    // Initialise alphahat = a, and V = P
    memcpy(alphahat, a, n * m * sizeof(double));
    memcpy(V, P, n * m * m * sizeof(double));

    // Smooth
    for (int t = n - 1; t >= 0; t--) {
      // Previous weighted sum of innovations (6.15.2)
      F77_CALL(dgemv)(
          tr, &m, &m,
          &pos, T, &m, r + ((t + 1) * m), &inc,
          &zero, r + (t * m), &inc, len_1);
      F77_CALL(dgemm)(
          tr, nt, &m, &m, &m,
          &pos, T, &m, N + ((t + 1) * m * m), &m,
          &zero, mm, &m, len_1, len_2);
      F77_CALL(dgemm)(
          nt, nt, &m, &m, &m,
          &pos, mm, &m, T, &m,
          &zero, N + (t * m * m), &m, len_1, len_2);

      if (disturbances) {
        // Smoothed eta disturbances (4.69)
        memcpy(V_eta + (t * q * q), Q, q * q * sizeof(double));
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
        // L = I - K z' is rank-1 (z = Z_i'), so it is never formed:
        //   L'r + F^-1 v z    = r + (F^-1 v - K'r) z
        //   L'NL + F^-1 z z'  = N - z w' - w z' + (s + F^-1) z z'
        // with w = N K and s = K'w
        double *K_ti = K + (t * m * p) + (i * m);
        double Kr = F77_CALL(ddot)(&m, K_ti, &inc, r + (t * m), &inc);
        F77_CALL(dsymv)(
          up, &m,
          &pos, N + (t * m * m), &m, K_ti, &inc,
          &zero, m1, &inc, len_1);
        double s = F77_CALL(ddot)(&m, K_ti, &inc, m1, &inc);

        if (disturbances) {
          // Smoothed epsilon disturbances
          epshat[(t * p) + i] =
            H[(i * p) + i] * (Finv[(t * p) + i] * v[(t * p) + i] - Kr);
          V_eps[(t * p) + i] = H[(i * p) + i] -
            H[(i * p) + i] * H[(i * p) + i] * (Finv[(t * p) + i] + s);
        }

        // Weighted sum of innovations (6.15.1)
        c = Finv[(t * p) + i] * v[(t * p) + i] - Kr;
        F77_CALL(daxpy)(&m, &c, Z + i, &p, r + (t * m), &inc);
        F77_CALL(dger)(
          &m, &m,
          &neg, Z + i, &p, m1, &inc,
          N + (t * m * m), &m);
        F77_CALL(dger)(
          &m, &m,
          &neg, m1, &inc, Z + i, &p,
          N + (t * m * m), &m);
        c = s + Finv[(t * p) + i];
        F77_CALL(dger)(
          &m, &m,
          &c, Z + i, &p, Z + i, &p,
          N + (t * m * m), &m);
      }

      if (smoother) {
        // Smoothed state (4.44)
        F77_CALL(dgemv)(
          nt, &m, &m,
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

  return 0;
}

extern "C" SEXP kfas0(
    SEXP y,
    SEXP Z,
    SEXP H,
    SEXP T,
    SEXP R,
    SEXP Q,
    SEXP a1,
    SEXP P1
) {
  SEXP dim_y = PROTECT(getAttrib(y, R_DimSymbol));
  SEXP dim_R = PROTECT(getAttrib(R, R_DimSymbol));
  
  int p = INTEGER(dim_y)[0];
  int n = INTEGER(dim_y)[1];
  int m = INTEGER(dim_R)[0];
  int q = INTEGER(dim_R)[1];

  SEXP v        = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP F        = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP att      = PROTECT(allocMatrix(REALSXP, m    , n));
  SEXP Ptt      = PROTECT(allocMatrix(REALSXP, m * m, n));
  SEXP a        = PROTECT(allocMatrix(REALSXP, m    , n + 1));
  SEXP P        = PROTECT(allocMatrix(REALSXP, m * m, n + 1));
  SEXP K        = PROTECT(allocMatrix(REALSXP, m * p, n));
  SEXP r        = PROTECT(allocMatrix(REALSXP, m    , n + 1));
  SEXP N        = PROTECT(allocMatrix(REALSXP, m * m, n + 1));
  SEXP epshat   = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP V_eps    = PROTECT(allocMatrix(REALSXP, p    , n));
  SEXP etahat   = PROTECT(allocMatrix(REALSXP, q    , n));
  SEXP V_eta    = PROTECT(allocMatrix(REALSXP, q * q, n));
  SEXP alphahat = PROTECT(allocMatrix(REALSXP, m    , n));
  SEXP V        = PROTECT(allocMatrix(REALSXP, m * m, n));

  int list_length = 15;

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

  memset(r_ptr + (n * m), 0, m * sizeof(double));
  memset(N_ptr + (n * m * m), 0, m * m * sizeof(double));

  if (kfas0_c(
    n,
    p,
    m,
    q,
    y_ptr,
    Z_ptr,
    H_ptr,
    T_ptr,
    R_ptr,
    Q_ptr,
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
    true,
    true,
    true
  ) != 0) printf("Filter failed!\n");

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

  UNPROTECT(18);

  return result;
}
