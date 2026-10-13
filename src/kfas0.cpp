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

// Dimensions and time strides of a validated model
typedef struct {
  int n;
  int p;
  int m;
  int q;
  int dZ;
  int dH;
  int dT;
  int dR;
  int dQ;
} model_dims;

static model_dims check_model(SEXP y, SEXP Z, SEXP H, SEXP T, SEXP R, SEXP Q,
                              SEXP a1, SEXP P1) {
  model_dims d;

  check_matrix(y, "y");
  check_matrix(R, "R");
  check_double(Z, -1, "Z");
  check_double(H, -1, "H");
  check_double(T, -1, "T");
  check_double(Q, -1, "Q");

  d.p = INTEGER(getAttrib(y, R_DimSymbol))[0];
  d.n = INTEGER(getAttrib(y, R_DimSymbol))[1];
  d.m = INTEGER(getAttrib(R, R_DimSymbol))[0];
  d.q = INTEGER(getAttrib(R, R_DimSymbol))[1];

  d.dZ = stride(Z, d.p * d.m, d.n, "Z");
  d.dH = stride(H, d.p * d.p, d.n, "H");
  d.dT = stride(T, d.m * d.m, d.n, "T");
  d.dR = stride(R, d.m * d.q, d.n, "R");
  d.dQ = stride(Q, d.q * d.q, d.n, "Q");
  check_double(a1, d.m, "a1");
  check_double(P1, (R_xlen_t) d.m * d.m, "P1");

  return d;
}

// The univariate treatment needs diagonal H. Otherwise each H_t (restricted
// to the observed elements of y_t) is factored as L_t L_t' and the model is
// transformed to y* = L^-1 y, Z* = L^-1 Z, H* = I.
typedef struct {
  bool diagonal;
  double *y;       // y*, or the original y if diagonal
  double *Z;
  double *H;
  int dZ;
  int dH;
  double *L;       // L_t at L + t * dL, leading dimension n_obs[t]
  int dL;
  int *n_obs;      // number of observed elements of y_t
  int *obs;        // their indices, at obs + t * p
} transformed_model;

static transformed_model transform_model(model_dims d, double *y, double *Z,
                                         double *H) {
  int n = d.n;
  int p = d.p;
  int m = d.m;
  int dZ = d.dZ;
  int dH = d.dH;

  // Scalars required for BLAS routines
  const char *lf = "l";
  const char *lo = "l";
  const char *nt = "n";
  const char *nu = "n";
  int inc = 1;
  double pos = 1.0;
  FC_LEN_T len_1 = 1;

  transformed_model tm;
  tm.diagonal = is_diagonal(H, p, dH ? n : 1);
  tm.y = y;
  tm.Z = Z;
  tm.H = H;
  tm.dZ = dZ;
  tm.dH = dH;
  tm.L = NULL;
  tm.dL = dH;
  tm.n_obs = NULL;
  tm.obs = NULL;
  if (tm.diagonal) return tm;

  // Observed elements of each y_t
  bool has_missing = false;
  tm.n_obs = (int *) R_alloc(n, sizeof(int));
  tm.obs = (int *) R_alloc((size_t) n * p, sizeof(int));
  for (ptrdiff_t t = 0; t < n; t++) {
    tm.n_obs[t] = 0;
    for (int i = 0; i < p; i++) {
      if (!ISNAN(y[(t * p) + i])) tm.obs[(t * p) + tm.n_obs[t]++] = i;
    }
    if (tm.n_obs[t] < p) has_missing = true;
  }

  if (!has_missing) {
    tm.L = (double *) R_alloc((size_t) p * p * (dH ? n : 1), sizeof(double));
    cholesky(H, p, dH ? n : 1, tm.L);

    tm.y = (double *) R_alloc((size_t) n * p, sizeof(double));
    memcpy(tm.y, y, (size_t) n * p * sizeof(double));
    for (ptrdiff_t t = 0; t < n; t++) {
      F77_CALL(dtrsv)(
        lo, nt, nu, &p,
        tm.L + (t * tm.dL), &p, tm.y + (t * p), &inc, len_1, len_1, len_1);
    }

    // Z* varies over time if either Z or H does
    tm.dZ = (dZ != 0 || dH != 0) ? p * m : 0;
    int nZ = tm.dZ ? n : 1;
    tm.Z = (double *) R_alloc((size_t) p * m * nZ, sizeof(double));
    for (ptrdiff_t k = 0; k < nZ; k++) {
      memcpy(tm.Z + (k * p * m), Z + (k * dZ), p * m * sizeof(double));
      F77_CALL(dtrsm)(
        lf, lo, nt, nu, &p, &m,
        &pos, tm.L + (k * tm.dL), &p, tm.Z + (k * p * m), &p,
        len_1, len_1, len_1, len_1);
    }
  } else {
    // With missing observations, transform each y_t using only its observed
    // elements: H_oo = L_t L_t' (stored compactly, leading dimension
    // n_obs[t]). Missing elements of y* stay NA and are skipped.
    tm.dL = p * p;
    tm.L = (double *) R_alloc((size_t) n * p * p, sizeof(double));
    tm.y = (double *) R_alloc((size_t) n * p, sizeof(double));
    fill_na(tm.y, (R_xlen_t) n * p);
    tm.dZ = p * m;
    tm.Z = (double *) S_alloc((size_t) n * p * m, sizeof(double));
    double *yc = (double *) R_alloc(p, sizeof(double));
    double *Zc = (double *) R_alloc(p * m, sizeof(double));
    int info;

    for (ptrdiff_t t = 0; t < n; t++) {
      int po = tm.n_obs[t];
      int *idx = tm.obs + (t * p);
      double *H_t = H + (t * dH);
      double *Z_t = Z + (t * dZ);
      double *L_t = tm.L + (t * tm.dL);
      if (po == 0) continue;

      for (int k = 0; k < po; k++) {
        for (int j = 0; j < po; j++) L_t[j + (k * po)] = H_t[idx[j] + (idx[k] * p)];
      }
      F77_CALL(dpotrf)(lo, &po, L_t, &po, &info, len_1);
      if (info != 0) {
        error("H is not positive definite at time %lld", (long long) t + 1);
      }

      for (int j = 0; j < po; j++) yc[j] = y[(t * p) + idx[j]];
      F77_CALL(dtrsv)(
        lo, nt, nu, &po,
        L_t, &po, yc, &inc, len_1, len_1, len_1);
      for (int j = 0; j < po; j++) tm.y[(t * p) + idx[j]] = yc[j];

      for (int c = 0; c < m; c++) {
        for (int j = 0; j < po; j++) Zc[j + (c * po)] = Z_t[idx[j] + (c * p)];
      }
      F77_CALL(dtrsm)(
        lf, lo, nt, nu, &po, &m,
        &pos, L_t, &po, Zc, &po,
        len_1, len_1, len_1, len_1);
      for (int c = 0; c < m; c++) {
        for (int j = 0; j < po; j++) {
          tm.Z[(t * p * m) + idx[j] + (c * p)] = Zc[j + (c * po)];
        }
      }
    }
  }

  tm.H = (double *) S_alloc(p * p, sizeof(double));
  for (int i = 0; i < p; i++) tm.H[(i * p) + i] = 1.0;
  tm.dH = 0;

  return tm;
}

// Jacobian of y* = L^-1 y: log f(y) = log f(y*) - sum_t log det L_t
static double log_det_L(transformed_model tm, int n) {
  double log_det = 0.0;

  if (tm.diagonal) return 0.0;
  for (ptrdiff_t t = 0; t < n; t++) {
    for (int j = 0; j < tm.n_obs[t]; j++) {
      log_det += log(tm.L[(t * tm.dL) + (j * tm.n_obs[t]) + j]);
    }
  }

  return log_det;
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
    bool store,
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
  double *Finv = (double *) R_alloc((size_t) (store ? n : 1) * p, sizeof(double));
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

  // Offsets between time steps of the filter outputs: one slot per step
  // when stored, otherwise a single slot that each step overwrites (used
  // for the log-likelihood only, which needs O(m^2) memory instead of
  // O(n m^2)). The smoother needs the stored outputs.
  ptrdiff_t s_p  = store ? p : 0;
  ptrdiff_t s_m  = store ? m : 0;
  ptrdiff_t s_mm = store ? (ptrdiff_t) m * m : 0;
  ptrdiff_t s_mp = store ? (ptrdiff_t) m * p : 0;

  // Log-likelihood (7.2): sum of log(F) + v^2 / F over informative
  // observations, with the constant added at the end
  double ll_sum = 0.0;
  ptrdiff_t n_obs = 0;
  
  // Filter
  for (ptrdiff_t t = 0; t < n; t++) {
    double *Z_t = Z + (t * dZ);
    double *H_t = H + (t * dH);
    double *T_t = T + (t * dT);

    memcpy(att + (t * s_m), a + (t * s_m), m * sizeof(double));
    memcpy(Ptt + (t * s_mm), P + (t * s_mm), m * m * sizeof(double));
    
    for (int i = 0; i < p; i++) {
      // A missing observation carries no information: no update, and
      // Finv = 0 makes the smoother pass r and N through unchanged
      if (ISNAN(y[(t * p) + i])) {
        v[(t * s_p) + i] = NA_REAL;
        F[(t * s_p) + i] = NA_REAL;
        Finv[(t * s_p) + i] = 0.0;
        fill_na(K + (t * s_mp) + (i * m), m);
        continue;
      }

      // Innovation (6.13)
      v[(t * s_p) + i] = y[(t * p) + i] - F77_CALL(ddot)(
        &m, Z_t + i, &p, att + (t * s_m), &inc);
      F[(t * s_p) + i] = H_t[(i * p) + i];

      // M = Ptt Z_i' (returned as K), F = H_i + Z_i M
      F77_CALL(dsymv)(
        up, &m,
        &pos, Ptt + (t * s_mm), &m, Z_t + i, &p,
        &zero, K + (t * s_mp) + (i * m), &inc, len_1);
      F[(t * s_p) + i] += F77_CALL(ddot)(
        &m, K + (t * s_mp) + (i * m), &inc, Z_t + i, &p);

      // F is numerically zero when it is within rounding error of the
      // terms summed to form it; the observation then carries no
      // information, so skip the update (Finv = 0 also makes the smoother
      // pass r and N through unchanged)
      double scale = fabs(H_t[(i * p) + i]);
      for (int j = 0; j < m; j++) {
        scale += fabs(Z_t[i + (j * p)] * K[(t * s_mp) + (i * m) + j]);
      }
      if (F[(t * s_p) + i] <= sqrt(DBL_EPSILON) * scale) {
        Finv[(t * s_p) + i] = 0.0;
        memset(K + (t * s_mp) + (i * m), 0, m * sizeof(double));
        continue;
      }

      Finv[(t * s_p) + i] = 1.0 / F[(t * s_p) + i];
      ll_sum += log(F[(t * s_p) + i]) +
        v[(t * s_p) + i] * v[(t * s_p) + i] * Finv[(t * s_p) + i];
      n_obs++;

      // Update state (6.12) with gain M / F: att += M v / F,
      // Ptt -= M M' / F
      c = v[(t * s_p) + i] * Finv[(t * s_p) + i];
      F77_CALL(daxpy)(
        &m, &c, K + (t * s_mp) + (i * m), &inc,
        att + (t * s_m), &inc);
      c = -Finv[(t * s_p) + i];
      F77_CALL(dger)(
        &m, &m,
        &c, K + (t * s_mp) + (i * m), &inc, K + (t * s_mp) + (i * m), &inc,
        Ptt + (t * s_mm), &m);
    }

    // Predict state (6.14)
    F77_CALL(dgemv)(
      nt, &m, &m,
      &pos, T_t, &m, att + (t * s_m), &inc,
      &zero, a + ((t + 1) * s_m), &inc, len_1);
    if (varying_RQ) compute_RQR(m, q, R + (t * dR), Q + (t * dQ), mq, RQR);
    memcpy(P + ((t + 1) * s_mm), RQR, m * m * sizeof(double));
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, T_t, &m, Ptt + (t * s_mm), &m,
      &zero, mm, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, tr, &m, &m, &m,
      &pos, mm, &m, T_t, &m,
      &pos, P + ((t + 1) * s_mm), &m, len_1, len_2);
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
  model_dims d = check_model(y, Z, H, T, R, Q, a1, P1);
  int n = d.n;
  int p = d.p;
  int m = d.m;
  int q = d.q;
  int dZ = d.dZ;
  int dH = d.dH;

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
  const char *lo = "l";
  const char *nt = "n";
  const char *nu = "n";
  const char *up = "u";
  int inc = 1;
  double pos = 1.0;
  double zero = 0.0;
  FC_LEN_T len_1 = 1;

  transformed_model tm = transform_model(d, y_ptr, Z_ptr, H_ptr);
  bool diagonal = tm.diagonal;
  double *L = tm.L;
  int dL = tm.dL;
  int *n_obs = tm.n_obs;
  int *obs = tm.obs;

  // Original-scale V_eps for non-diagonal H needs the smoothed V
  bool run_smoother = do_smoother || (do_disturbances && !diagonal);

  kfas0_c(
    n,
    p,
    m,
    q,
    tm.y,
    tm.Z,
    tm.H,
    T_ptr,
    R_ptr,
    Q_ptr,
    tm.dZ,
    tm.dH,
    d.dT,
    d.dR,
    d.dQ,
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
    true,
    run_smoother,
    do_disturbances
  );
  REAL(loglik)[0] -= log_det_L(tm, n);

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

  UNPROTECT(17);

  return result;
}

// Log-likelihood only: the filter without storing per-step outputs, so it
// needs O(m^2) memory instead of O(n m^2)
extern "C" SEXP kfas0_loglik(
    SEXP y,
    SEXP Z,
    SEXP H,
    SEXP T,
    SEXP R,
    SEXP Q,
    SEXP a1,
    SEXP P1
) {
  model_dims d = check_model(y, Z, H, T, R, Q, a1, P1);
  int p = d.p;
  int m = d.m;

  transformed_model tm = transform_model(d, REAL(y), REAL(Z), REAL(H));

  // Single-step filter buffers
  double *v   = (double *) R_alloc(p,     sizeof(double));
  double *F   = (double *) R_alloc(p,     sizeof(double));
  double *att = (double *) R_alloc(m,     sizeof(double));
  double *Ptt = (double *) R_alloc(m * m, sizeof(double));
  double *a   = (double *) R_alloc(m,     sizeof(double));
  double *P   = (double *) R_alloc(m * m, sizeof(double));
  double *K   = (double *) R_alloc(m * p, sizeof(double));
  double loglik;

  memcpy(a, REAL(a1), m * sizeof(double));
  memcpy(P, REAL(P1), m * m * sizeof(double));

  kfas0_c(
    d.n,
    p,
    m,
    d.q,
    tm.y,
    tm.Z,
    tm.H,
    REAL(T),
    REAL(R),
    REAL(Q),
    tm.dZ,
    tm.dH,
    d.dT,
    d.dR,
    d.dQ,
    v,
    F,
    att,
    Ptt,
    a,
    P,
    K,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    &loglik,
    false,
    false,
    false
  );

  return ScalarReal(loglik - log_det_L(tm, d.n));
}
