// Kalman Filter (univariate treatment)
// Byron Botha 2025

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/RS.h>

static int epssmooth(
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
  char *tr = "t";
  char *nt = "n";
  int inc = 1;
  double pos = 1.0;
  double neg = -1.0;
  double zero = 0.0;

  // String lengths for cstring args
  FC_LEN_T len_1 = 1; 
  FC_LEN_T len_2 = 1;

  // Temporary storage for intermediate results
  double *Finv = malloc(n * p * sizeof(double));
  double *L    = malloc(m * m * sizeof(double));
  double *m1   = malloc(m * sizeof(double));
  double *mq   = malloc(m * q * sizeof(double));
  double *mm   = malloc(m * m * sizeof(double));
  double *qm_1 = malloc(q * m * sizeof(double));
  double *qm_2 = mq;
  double c;

  // Create mxm identity matrix
  double *Ident = calloc(m * m, sizeof(double));
  for (int i = 0; i < m; i++) Ident[(i * m) + i] = 1.0;

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
        F77_CALL(dgemm)(
          nt, nt, &inc, &m, &m,
          &pos, Z + i, &p, Ptt + (t * m * m), &m,
          &zero, m1, &m, len_1, len_2);
        F77_CALL(dgemv)(
          nt, &inc, &m,
          &pos, m1, &inc, Z + i, &p,
          &pos, F + (t * p) + i, &inc, len_1);
        
        // K matrix (6.13)
        Finv[(t * p) + i] = 1.0 / F[(t * p) + i];
        F77_CALL(dgemv)(
          nt, &m, &m,
          Finv + (t * p) + i, Ptt + (t * m * m), &m, Z + i, &p,
          &zero, K + (t * m * p) + (i * m), &inc, len_1);

        // Update state (6.12)
        F77_CALL(dgemv)(
          nt, &m, &inc,
          &pos, K + (t * m * p) + (i * m), &p, v + (t * p) + i, &inc,
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
      F77_CALL(dgemm)(
        nt, nt, &m, &q, &q,
        &pos, R, &m, Q, &q,
        &zero, mq, &m, len_1, len_2);
      F77_CALL(dgemm)(
        nt, tr, &m, &m, &q,
        &pos, mq, &m, R, &q,
        &zero, P + ((t + 1) * m * m), &m, len_1, len_2);
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

    // Initialise epshat = v, and V_eps = F
    memcpy(epshat, v, n * p * sizeof(double));
    memcpy(V_eps, Finv, n * p * sizeof(double));

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
        F77_CALL(dgemm)(
          nt, tr, &q, &m, &q,
          &pos, Q, &q, R, &m,
          &zero, qm_1, &q, len_1, len_2);
        F77_CALL(dgemv)(
          nt, &q, &m,
          &pos, qm_1, &q, r + ((t + 1) * m), &inc,
          &pos, etahat + (t * q), &inc, len_1);
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
        if (disturbances) {
          // Smoothed epsilon disturbances
          F77_CALL(dgemv)(
            tr, &m, &inc,
            &neg, K + (t * m * p) + (i * m), &m, r + (t * m), &inc,
            Finv + (t * p) + i, epshat + (t * p) + i, &inc, len_1);
          epshat[(t * p) + i] *= H[(i * p) + i];
          F77_CALL(dgemv)(
            nt, &m, &m,
            &pos, N + (t * m * m), &m, K + (t * m * p) + (i * m), &inc,
            &zero, m1, &inc, len_1);
          F77_CALL(dgemv)(
            tr, &m, &inc,
            &pos, K + (t * m * p) + (i * m), &m, m1, &inc,
            &pos, V_eps + (t * p) + i, &inc, len_1);
          V_eps[(t * p) + i] *= H[(i * p) + i] * H[(i * p) + i];
          V_eps[(t * p) + i] = H[(i * p) + i] - V_eps[(t * p) + i];
        }

        // L matrix
        memcpy(L, Ident, m * m * sizeof(double));
        F77_CALL(dgemm)(
          nt, nt, &m, &m, &inc,
          &neg, K + (t * m * p) + (i * m), &m, Z + i, &p,
          &pos, L, &m, len_1, len_2);

        // Weighted sum of innovations (6.15.1)
        memcpy(m1, r + (t * m), m * sizeof(double));
        F77_CALL(dgemv)(
          tr, &m, &m,
          &pos, L, &m, m1, &inc,
          &zero, r + (t * m), &inc, len_1);
        F77_CALL(dgemv)(
          tr, &inc, &m,
          Finv + (t * p) + i, Z + i, &p, v + (t * p) + i, &inc,
          &pos, r + (t * m), &inc, len_1);
        F77_CALL(dgemm)(
          tr, nt, &m, &m, &m,
          &pos, L, &m, N + (t * m * m), &m,
          &zero, mm, &m, len_1, len_2);
        F77_CALL(dgemm)(
          nt, nt, &m, &m, &m,
          &pos, mm, &m, L, &m,
          &zero, N + (t * m * m), &m, len_1, len_2);
        F77_CALL(dgemm)(
          tr, nt, &m, &m, &inc,
          Finv + (t * p) + i, Z + i, &p, Z + i, &p,
          &pos, N + (t * m * m), &m, len_1, len_2);
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

  free(Ident);
  free(Finv);
  free(L);
  free(m1);
  free(mq);
  free(mm);
  free(qm_1);

  return 0;
}

// int simsmooth()

SEXP kfas0(
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

  if (epssmooth(
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
