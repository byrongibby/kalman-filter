// Kalman Filter (multivariate treatment)
// Byron Botha 2025

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "kfas_utils.h"

// Filter and smoother with y_t processed as a vector (Durbin & Koopman,
// sections 4.3-4.5). H may be non-diagonal. Outputs are laid out as in
// kfas0_c, except F is p x p per time step and K is P_t Z_t'.
static void kfas_c(
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
  const char *rt = "r";
  int inc = 1;
  int info;
  double pos = 1.0;
  double neg = -1.0;
  double zero = 0.0;

  // String lengths for cstring args
  FC_LEN_T len_1 = 1;
  FC_LEN_T len_2 = 1;

  // Kept from the filter for the smoother
  // (R_alloc memory is reclaimed by R when the .Call returns)
  double *U     = (double *) R_alloc((size_t) n * p * p, sizeof(double));
  double *Finvv = (double *) R_alloc((size_t) n * p,     sizeof(double));
  double *FinvZ = (double *) R_alloc((size_t) n * p * m, sizeof(double));
  double *Kg    = (double *) R_alloc((size_t) n * m * p, sizeof(double));

  // Temporary storage for intermediate results
  double *G    = (double *) R_alloc(p * m, sizeof(double));
  double *L    = (double *) R_alloc(m * m, sizeof(double));
  double *mm   = (double *) R_alloc(m * m, sizeof(double));
  double *mp   = (double *) R_alloc(m * p, sizeof(double));
  double *pp_1 = (double *) R_alloc(p * p, sizeof(double));
  double *pp_2 = (double *) R_alloc(p * p, sizeof(double));
  double *p1   = (double *) R_alloc(p,     sizeof(double));
  double *mq   = (double *) R_alloc(m * q, sizeof(double));
  double *qm_1 = (double *) R_alloc(q * m, sizeof(double));
  double *qm_2 = mq;
  double *RQR  = (double *) R_alloc(m * m, sizeof(double));

  // Observed elements of each y_t: count n_obs[t] and indices obs[t * p + j].
  // Quantities kept for the smoother are stored compactly for the observed
  // sub-vector (leading dimension n_obs[t]).
  int *n_obs   = (int *) R_alloc(n, sizeof(int));
  int *obs     = (int *) R_alloc((size_t) n * p, sizeof(int));
  double *yc   = (double *) R_alloc(p,     sizeof(double));
  double *Zc   = (double *) R_alloc(p * m, sizeof(double));
  double *Hc   = (double *) R_alloc(p * p, sizeof(double));
  double *Mc   = (double *) R_alloc(m * p, sizeof(double));

  // RQR' and QR' are computed once unless R or Q vary over time
  bool varying_RQ = (dR != 0 || dQ != 0);
  if (!varying_RQ) {
    compute_RQR(m, q, R, Q, mq, RQR);
    compute_QR(m, q, R, Q, qm_1);
  }

  // Log-likelihood (7.2): sum of log det F + v' F^-1 v over the observed
  // sub-vectors, with the constant added at the end
  double ll_sum = 0.0;
  ptrdiff_t n_total = 0;

  // Filter
  for (ptrdiff_t t = 0; t < n; t++) {
    double *Z_t   = Z + (t * dZ);
    double *H_t   = H + (t * dH);
    double *T_t   = T + (t * dT);
    double *a_t   = a + (t * m);
    double *P_t   = P + (t * m * m);
    double *v_t   = v + (t * p);
    double *F_t   = F + (t * p * p);
    double *K_t   = K + (t * m * p);
    double *U_t   = U + (t * p * p);
    double *w_t   = Finvv + (t * p);
    double *FZ_t  = FinvZ + (t * p * m);
    double *Kg_t  = Kg + (t * m * p);
    double *att_t = att + (t * m);
    double *Ptt_t = Ptt + (t * m * m);
    int *idx = obs + (t * p);

    int po = 0;
    for (int i = 0; i < p; i++) if (!ISNAN(y[(t * p) + i])) idx[po++] = i;
    n_obs[t] = po;
    n_total += po;

    // v, F and K are NA at missing observations
    fill_na(v_t, p);
    fill_na(F_t, p * p);
    fill_na(K_t, m * p);

    memcpy(att_t, a_t, m * sizeof(double));
    memcpy(Ptt_t, P_t, m * m * sizeof(double));

    if (po > 0) {
      // Observed sub-vector y_o, Z_o, H_oo
      for (int j = 0; j < po; j++) yc[j] = y[(t * p) + idx[j]];
      for (int c = 0; c < m; c++) {
        for (int j = 0; j < po; j++) Zc[j + (c * po)] = Z_t[idx[j] + (c * p)];
      }
      for (int k = 0; k < po; k++) {
        for (int j = 0; j < po; j++) Hc[j + (k * po)] = H_t[idx[j] + (idx[k] * p)];
      }

      // Innovation v = y - Z a (4.24)
      F77_CALL(dgemv)(
        nt, &po, &m,
        &neg, Zc, &po, a_t, &inc,
        &pos, yc, &inc, len_1);
      for (int j = 0; j < po; j++) v_t[idx[j]] = yc[j];

      // M = P Z' (returned as K), F = Z M + H
      F77_CALL(dgemm)(
        nt, tr, &m, &po, &m,
        &pos, P_t, &m, Zc, &po,
        &zero, Mc, &m, len_1, len_2);
      for (int j = 0; j < po; j++) {
        memcpy(K_t + (idx[j] * m), Mc + (j * m), m * sizeof(double));
      }
      memcpy(U_t, Hc, po * po * sizeof(double));
      F77_CALL(dgemm)(
        nt, nt, &po, &po, &m,
        &pos, Zc, &po, Mc, &m,
        &pos, U_t, &po, len_1, len_2);
      for (int k = 0; k < po; k++) {
        for (int j = 0; j < po; j++) F_t[idx[j] + (idx[k] * p)] = U_t[j + (k * po)];
      }

      // F = U'U (Cholesky), then F^-1 v and F^-1 Z
      F77_CALL(dpotrf)(up, &po, U_t, &po, &info, len_1);
      if (info != 0) {
        error("F is not positive definite at time %lld; observations with "
              "no information are only supported by the univariate "
              "treatment", (long long) t + 1);
      }
      memcpy(w_t, yc, po * sizeof(double));
      F77_CALL(dpotrs)(up, &po, &inc, U_t, &po, w_t, &po, &info, len_1);
      memcpy(FZ_t, Zc, po * m * sizeof(double));
      F77_CALL(dpotrs)(up, &po, &m, U_t, &po, FZ_t, &po, &info, len_1);

      // log det F = 2 sum log diag(U)
      for (int j = 0; j < po; j++) ll_sum += 2.0 * log(U_t[(j * po) + j]);
      ll_sum += F77_CALL(ddot)(&po, yc, &inc, w_t, &inc);

      // G = F^-1 Z P = (M F^-1)'
      F77_CALL(dsymm)(
        rt, up, &po, &m,
        &pos, P_t, &m, FZ_t, &po,
        &zero, G, &po, len_1, len_1);

      // Update (4.26): att = a + M F^-1 v, Ptt = P - M F^-1 M'
      F77_CALL(dgemv)(
        tr, &po, &m,
        &pos, G, &po, yc, &inc,
        &pos, att_t, &inc, len_1);
      F77_CALL(dgemm)(
        nt, nt, &m, &m, &po,
        &neg, Mc, &m, G, &po,
        &pos, Ptt_t, &m, len_1, len_2);

      // Kalman gain K = T M F^-1 (4.24), kept for the smoother
      F77_CALL(dgemm)(
        nt, tr, &m, &po, &m,
        &pos, T_t, &m, G, &po,
        &zero, Kg_t, &m, len_1, len_2);
    }

    // Predict (4.27)
    F77_CALL(dgemv)(
      nt, &m, &m,
      &pos, T_t, &m, att_t, &inc,
      &zero, a + ((t + 1) * m), &inc, len_1);
    if (varying_RQ) compute_RQR(m, q, R + (t * dR), Q + (t * dQ), mq, RQR);
    memcpy(P + ((t + 1) * m * m), RQR, m * m * sizeof(double));
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, T_t, &m, Ptt_t, &m,
      &zero, mm, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &m, &m, &m,
      &pos, mm, &m, T_t, &m,
      &pos, P + ((t + 1) * m * m), &m, len_1, len_2);
  }

  *loglik = -0.5 * ((double) n_total * LOG_2PI + ll_sum);

  if (!smoother && !disturbances) return;

  if (smoother) {
    // Initialise alphahat = a, and V = P
    memcpy(alphahat, a, (size_t) n * m * sizeof(double));
    memcpy(V, P, (size_t) n * m * m * sizeof(double));
  }

  // Smooth. r and N at index t + 1 are r_t and N_t in Durbin & Koopman's
  // notation; index t receives r_t-1 and N_t-1.
  for (ptrdiff_t t = n - 1; t >= 0; t--) {
    double *Z_t    = Z + (t * dZ);
    double *H_t    = H + (t * dH);
    double *T_t    = T + (t * dT);
    double *Q_t    = Q + (t * dQ);
    double *P_t    = P + (t * m * m);
    double *U_t    = U + (t * p * p);
    double *w_t    = Finvv + (t * p);
    double *FZ_t   = FinvZ + (t * p * m);
    double *Kg_t   = Kg + (t * m * p);
    double *r_t    = r + (t * m);
    double *r_next = r + ((t + 1) * m);
    double *N_t    = N + (t * m * m);
    double *N_next = N + ((t + 1) * m * m);
    int *idx = obs + (t * p);
    int po = n_obs[t];

    for (int c = 0; c < m; c++) {
      for (int j = 0; j < po; j++) Zc[j + (c * po)] = Z_t[idx[j] + (c * p)];
    }

    if (disturbances) {
      // Smoothed eta disturbances (4.69)
      if (varying_RQ) compute_QR(m, q, R + (t * dR), Q_t, qm_1);
      memcpy(V_eta + (t * q * q), Q_t, q * q * sizeof(double));
      F77_CALL(dgemv)(
        nt, &q, &m,
        &pos, qm_1, &q, r_next, &inc,
        &zero, etahat + (t * q), &inc, len_1);
      F77_CALL(dgemm)(
        nt, nt, &q, &m, &m,
        &pos, qm_1, &q, N_next, &m,
        &zero, qm_2, &q, len_1, len_2);
      F77_CALL(dgemm)(
        nt, tr, &q, &q, &m,
        &neg, qm_2, &q, qm_1, &q,
        &pos, V_eta + (t * q * q), &q, len_1, len_2);

      // Smoothed epsilon disturbances (4.69) for all p elements, observed
      // or not, with W selecting the observed ones:
      //   epshat = H W' d,  d = F^-1 v - K' r
      //   Var(eps | y) = H - H W' (F^-1 + K' N K) W H  (diagonal only)
      if (po == 0) {
        for (int i = 0; i < p; i++) {
          epshat[(t * p) + i] = 0.0;
          V_eps[(t * p) + i] = H_t[(i * p) + i];
        }
      } else {
        // Hc = H W' (p x po)
        for (int j = 0; j < po; j++) {
          memcpy(Hc + (j * p), H_t + (idx[j] * p), p * sizeof(double));
        }
        memcpy(p1, w_t, po * sizeof(double));
        F77_CALL(dgemv)(
          tr, &m, &po,
          &neg, Kg_t, &m, r_next, &inc,
          &pos, p1, &inc, len_1);
        F77_CALL(dgemv)(
          nt, &p, &po,
          &pos, Hc, &p, p1, &inc,
          &zero, epshat + (t * p), &inc, len_1);

        // dpotri sets the upper triangle of F^-1; mirror it to the lower
        memcpy(pp_1, U_t, po * po * sizeof(double));
        F77_CALL(dpotri)(up, &po, pp_1, &po, &info, len_1);
        for (int j = 0; j < po; j++) {
          for (int i = j + 1; i < po; i++) {
            pp_1[i + (j * po)] = pp_1[j + (i * po)];
          }
        }
        F77_CALL(dgemm)(
          nt, nt, &m, &po, &m,
          &pos, N_next, &m, Kg_t, &m,
          &zero, mp, &m, len_1, len_2);
        F77_CALL(dgemm)(
          tr, nt, &po, &po, &m,
          &pos, Kg_t, &m, mp, &m,
          &pos, pp_1, &po, len_1, len_2);
        F77_CALL(dgemm)(
          nt, nt, &p, &po, &po,
          &pos, Hc, &p, pp_1, &po,
          &zero, pp_2, &p, len_1, len_2);
        for (int i = 0; i < p; i++) {
          V_eps[(t * p) + i] = H_t[(i * p) + i] -
            F77_CALL(ddot)(&po, pp_2 + i, &p, Hc + i, &p);
        }
      }
    }

    // L = T - K Z (L = T when nothing is observed)
    memcpy(L, T_t, m * m * sizeof(double));
    if (po > 0) {
      F77_CALL(dgemm)(
        nt, nt, &m, &m, &po,
        &neg, Kg_t, &m, Zc, &po,
        &pos, L, &m, len_1, len_2);
    }

    // r_t-1 = Z' F^-1 v + L' r_t (4.38)
    F77_CALL(dgemv)(
      tr, &m, &m,
      &pos, L, &m, r_next, &inc,
      &zero, r_t, &inc, len_1);
    if (po > 0) {
      F77_CALL(dgemv)(
        tr, &po, &m,
        &pos, Zc, &po, w_t, &inc,
        &pos, r_t, &inc, len_1);
    }

    // N_t-1 = Z' F^-1 Z + L' N_t L (4.42)
    F77_CALL(dgemm)(
      tr, nt, &m, &m, &m,
      &pos, L, &m, N_next, &m,
      &zero, mm, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, mm, &m, L, &m,
      &zero, N_t, &m, len_1, len_2);
    if (po > 0) {
      F77_CALL(dgemm)(
        tr, nt, &m, &m, &po,
        &pos, Zc, &po, FZ_t, &po,
        &pos, N_t, &m, len_1, len_2);
    }

    if (smoother) {
      // Smoothed state (4.39, 4.43)
      F77_CALL(dsymv)(
        up, &m,
        &pos, P_t, &m, r_t, &inc,
        &pos, alphahat + (t * m), &inc, len_1);
      F77_CALL(dgemm)(
        nt, nt, &m, &m, &m,
        &pos, P_t, &m, N_t, &m,
        &zero, mm, &m, len_1, len_2);
      F77_CALL(dgemm)(
        nt, nt, &m, &m, &m,
        &neg, mm, &m, P_t, &m,
        &pos, V + (t * m * m), &m, len_1, len_2);
    }
  }
}

extern "C" SEXP kfas(
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
  SEXP F        = PROTECT(alloc3DArray(REALSXP, p, p, n));
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

  double *r_ptr = REAL(r);
  double *N_ptr = REAL(N);

  memcpy(REAL(a), REAL(a1), m * sizeof(double));
  memcpy(REAL(P), REAL(P1), m * m * sizeof(double));

  memset(r_ptr + ((ptrdiff_t) n * m), 0, m * sizeof(double));
  memset(N_ptr + ((ptrdiff_t) n * m * m), 0, m * m * sizeof(double));

  kfas_c(
    n,
    p,
    m,
    q,
    REAL(y),
    REAL(Z),
    REAL(H),
    REAL(T),
    REAL(R),
    REAL(Q),
    dZ,
    dH,
    dT,
    dR,
    dQ,
    REAL(v),
    REAL(F),
    REAL(att),
    REAL(Ptt),
    REAL(a),
    REAL(P),
    REAL(K),
    r_ptr,
    N_ptr,
    REAL(epshat),
    REAL(V_eps),
    REAL(etahat),
    REAL(V_eta),
    REAL(alphahat),
    REAL(V),
    REAL(loglik),
    do_smoother,
    do_disturbances
  );

  // Outputs that were not requested are NA rather than uninitialised
  if (!do_smoother) {
    fill_na(REAL(alphahat), XLENGTH(alphahat));
    fill_na(REAL(V), XLENGTH(V));
  }
  if (!do_disturbances) {
    fill_na(REAL(epshat), XLENGTH(epshat));
    fill_na(REAL(V_eps), XLENGTH(V_eps));
    fill_na(REAL(etahat), XLENGTH(etahat));
    fill_na(REAL(V_eta), XLENGTH(V_eta));
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
