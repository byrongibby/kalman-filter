// Kalman Filter
// Byron Botha 2025

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

static int filter(
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
    double *K
) {
  // Parameters required by BLAS/LAPACK
  char *tr = "t";
  char *nt = "n";
  int inc = 1;
  double pos = 1.0;
  double neg = -1.0;
  double zero = 0.0;
  int *ipiv = malloc(p * sizeof(int));
  int info = 0;

  // Temporary storage for intermediate results
  double *ZP = malloc(p * m * sizeof(double));
  double *FinvZP = malloc(p * m * sizeof(double));
  double *RQ = malloc(m * q * sizeof(double));
  double *TPtt = malloc(m * m * sizeof(double));

  // String lengths for cstring args
  FC_LEN_T len_1 = 1; 
  FC_LEN_T len_2 = 1;

  for (int t = 0; t < n; t++) {
    // Innovate
    memcpy(v + (t * p), y + (t * p), p * sizeof(double));
    memcpy(F + (t * p) * p, H, p * p * sizeof(double));
    F77_CALL(dgemv)(
      nt, &p, &m,
      &neg, Z, &p, a + (t * m), &inc,
      &pos, v + (t * p), &inc, len_1);
    F77_CALL(dgemm)(
      nt, nt, &p, &m, &m,
      &pos, Z, &p, P + (t * m) * m, &m,
      &zero, ZP, &p, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &p, &p, &m,
      &pos, ZP, &p, Z, &p,
      &pos, F + (t * p) * p, &p, len_1, len_2);

    // Update
    memcpy(FinvZP, ZP, p * m * sizeof(double));
    memcpy(att + (t * m), a + (t * m), m * sizeof(double));
    memcpy(Ptt + (t * m * m), P + (t * m) * m, m * m * sizeof(double));
    F77_CALL(dgesv)(
      &p, &m,
      F + (t * p) * p, &p, ipiv,
      FinvZP, &p, &info);
    if (info != 0) return -1;
    F77_CALL(dgemv)(
      tr, &p, &m,
      &pos, FinvZP, &p, v + (t * p), &inc,
      &pos, att + (t * m), &inc, len_1);
    F77_CALL(dgemm)(
      tr, nt, &m, &m, &p,
      &neg, FinvZP, &p, ZP, &p,
      &pos, Ptt + (t * m) * m, &m, len_1, len_2);
    
    // Kalman gain
    F77_CALL(dgemm)(
      nt, tr, &m, &p, &m,
      &pos, T, &m, FinvZP, &p,
      &zero, K + (t * m) * p, &m, len_1, len_2);

    // Predict
    F77_CALL(dgemv)(
      nt, &m, &m,
      &pos, T, &m, att + (t * m), &inc,
      &zero, a + ((t + 1) * m), &inc, len_1);
    F77_CALL(dgemm)(
      nt, nt, &m, &q, &q,
      &pos, R, &m, Q, &q,
      &zero, RQ, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &m, &m, &q,
      &pos, RQ, &m, R, &q,
      &zero, P + ((t + 1) * m) * m, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, T, &m, Ptt + (t * m) * m, &m,
      &zero, TPtt, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &m, &m, &m,
      &pos, TPtt, &m, T, &m,
      &pos, P + ((t + 1) * m) * m, &m, len_1, len_2);
  }

  free(ipiv);
  free(ZP);
  free(FinvZP);
  free(RQ);
  free(TPtt);

  return 0;
}

static int smooth(
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
    double *alphahat,
    double *V
) {
  // Parameters required by BLAS/LAPACK
  char *tr = "t";
  char *nt = "n";
  int inc = 1;
  double pos = 1.0;
  double neg = -1.0;
  double zero = 0.0;
  int *ipiv = malloc(p * sizeof(int));
  int info = 0;

  // Temporary storage for intermediary results (filter)
  double *ZP = malloc(p * m * sizeof(double));
  double *FinvZ = malloc((p * m) * n * sizeof(double));
  double *FinvZP = malloc(p * m * sizeof(double));
  double *RQ = malloc(m * q * sizeof(double));
  double *TPtt = malloc(m * m * sizeof(double));

  // Temporary storage for intermediary results (smoother)
  double *L = malloc(m * m * sizeof(double));
  double *LN = malloc(m * m * sizeof(double));
  double *PN = malloc(m * m * sizeof(double));

  // String lengths for cstring args
  FC_LEN_T len_1 = 1; 
  FC_LEN_T len_2 = 1;

  for (int t = 0; t < n; t++) {
    // Innovate
    memcpy(v + (t * p), y + (t * p), p * sizeof(double));
    memcpy(F + (t * p) * p, H, p * p * sizeof(double));
    F77_CALL(dgemv)(
      nt, &p, &m,
      &neg, Z, &p, a + (t * m), &inc,
      &pos, v + (t * p), &inc, len_1);
    F77_CALL(dgemm)(
      nt, nt, &p, &m, &m,
      &pos, Z, &p, P + (t * m) * m, &m,
      &zero, ZP, &p, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &p, &p, &m,
      &pos, ZP, &p, Z, &p,
      &pos, F + (t * p) * p, &p, len_1, len_2);

    // Update
    memcpy(FinvZ + (t * p) * m, Z, p * m * sizeof(double));
    memcpy(att + (t * m), a + (t * m), m * sizeof(double));
    memcpy(Ptt + (t * m * m), P + (t * m) * m, m * m * sizeof(double));
    F77_CALL(dgesv)(
      &p, &m,
      F + (t * p) * p, &p, ipiv,
      FinvZ + (t * p) * m, &p, &info);
    if (info != 0) return -1;
    F77_CALL(dgemm)(
      nt, nt, &p, &m, &m,
      &pos, FinvZ + (t * p) * m, &p, P + (t * m) * m, &m,
      &zero, FinvZP, &p, len_1, len_2);
    F77_CALL(dgemv)(
      tr, &p, &m,
      &pos, FinvZP, &p, v + (t * p), &inc,
      &pos, att + (t * m), &inc, len_1);
    F77_CALL(dgemm)(
      tr, nt, &m, &m, &p,
      &neg, FinvZP, &p, ZP, &p,
      &pos, Ptt + (t * m) * m, &m, len_1, len_2);
    
    // Kalman gain
    F77_CALL(dgemm)(
      nt, tr, &m, &p, &m,
      &pos, T, &m, FinvZP, &p,
      &zero, K + (t * m) * p, &m, len_1, len_2);

    // Predict
    F77_CALL(dgemv)(
      nt, &m, &m,
      &pos, T, &m, att + (t * m), &inc,
      &zero, a + ((t + 1) * m), &inc, len_1);
    F77_CALL(dgemm)(
      nt, nt, &m, &q, &q,
      &pos, R, &m, Q, &q,
      &zero, RQ, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &m, &m, &q,
      &pos, RQ, &m, R, &q,
      &zero, P + ((t + 1) * m) * m, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, T, &m, Ptt + (t * m) * m, &m,
      &zero, TPtt, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &m, &m, &m,
      &pos, TPtt, &m, T, &m,
      &pos, P + ((t + 1) * m) * m, &m, len_1, len_2);
  }

  // Smooth
  for (int t = n - 1; t >= 0; t--) {
    memcpy(L, T, m * m * sizeof(double));
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &p,
      &neg, K + (t * m) * p, &m, Z, &p,
      &pos, L, &m, len_1, len_2);

    F77_CALL(dgemv)(
      tr, &m, &m,
      &pos, L, &m, r + ((t + 1) * m), &inc,
      &zero, r + (t * m), &inc, len_1);
    F77_CALL(dgemv)(
      tr, &p, &m,
      &pos, FinvZ + (t * p) * m, &p, v + (t * m), &inc,
      &pos, r + (t * m), &inc, len_1);
    F77_CALL(dgemm)(
      tr, nt, &m, &m, &p,
      &pos, FinvZ + (t * p) * m, &p, Z, &p,
      &zero, N + (t * m) * m, &m, len_1, len_2);
    F77_CALL(dgemm)(
      tr, nt, &m, &m, &m,
      &pos, L, &m, N + ((t + 1) * m) * m, &m,
      &zero, LN, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, LN, &m, L, &m,
      &pos, N + (t * m) * m, &m, len_1, len_2);

    memcpy(alphahat + (t * m), a + (t * m), m * sizeof(double));
    memcpy(V + (t * m) * m, P + (t * m) * m, m * m * sizeof(double));
    F77_CALL(dgemv)(
      nt, &m, &m,
      &pos, P + (t * m) * m, &m, r + (t * m), &inc,
      &pos, alphahat + (t * m), &inc, len_1);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, P + (t * m) * m, &m, N + (t * m) * m, &m,
      &zero, PN, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &neg, PN, &m, P + (t * m) * m, &m,
      &pos, V + (t * m) * m, &m, len_1, len_2);
  }

  free(ipiv);
  free(ZP);
  free(FinvZ);
  free(FinvZP);
  free(RQ);
  free(TPtt);
  free(L);
  free(LN);
  free(PN);

  return 0;
}

SEXP kfas(
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
  SEXP F        = PROTECT(allocMatrix(REALSXP, p * p, n));
  SEXP att      = PROTECT(allocMatrix(REALSXP, m    , n));
  SEXP Ptt      = PROTECT(allocMatrix(REALSXP, m * m, n));
  SEXP a        = PROTECT(allocMatrix(REALSXP, m    , n + 1));
  SEXP P        = PROTECT(allocMatrix(REALSXP, m * m, n + 1));
  SEXP K        = PROTECT(allocMatrix(REALSXP, m * p, n));
  SEXP r        = PROTECT(allocMatrix(REALSXP, m    , n + 1));
  SEXP N        = PROTECT(allocMatrix(REALSXP, m * m, n + 1));
  SEXP alphahat = PROTECT(allocMatrix(REALSXP, m    , n));
  SEXP V        = PROTECT(allocMatrix(REALSXP, m * m, n));

  int list_length = 11;

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
  double *alphahat_ptr = REAL(alphahat);
  double *V_ptr        = REAL(V);

  memcpy(a_ptr, a1_ptr, m * sizeof(double));
  memcpy(P_ptr, P1_ptr, m * m * sizeof(double));

  memset(r_ptr + n * m, 0, m * sizeof(double));
  memset(N_ptr + n * (m * m), 0, m * m * sizeof(double));

  if (smooth(
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
    alphahat_ptr,
    V_ptr
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
  SET_VECTOR_ELT(result,  9, alphahat);
  SET_VECTOR_ELT(result, 10, V);

  UNPROTECT(14);

  return result;
}
