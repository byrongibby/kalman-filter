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
#include "kfas_model.h"

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

model_dims check_model(SEXP y, SEXP Z, SEXP H, SEXP T, SEXP R, SEXP Q,
                       SEXP a1, SEXP P1, SEXP P1inf, SEXP dvec, SEXP cvec) {
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
  check_double(P1inf, (R_xlen_t) d.m * d.m, "P1inf");

  // Intercepts are optional (NULL)
  d.dd = 0;
  d.dc = 0;
  if (!isNull(dvec)) {
    check_double(dvec, -1, "d");
    d.dd = stride(dvec, d.p, d.n, "d");
  }
  if (!isNull(cvec)) {
    check_double(cvec, -1, "c");
    d.dc = stride(cvec, d.m, d.n, "c");
  }

  return d;
}

bool any_nonzero(double *x, R_xlen_t len) {
  for (R_xlen_t i = 0; i < len; i++) if (x[i] != 0.0) return true;
  return false;
}

// Observation intercept: the filter runs on y - d, so v = y - d - Z a.
// Returns y itself if there is no intercept.
double *subtract_intercept(model_dims d, double *y, SEXP dvec) {
  if (isNull(dvec)) return y;

  double *dv = REAL(dvec);
  double *yd = (double *) R_alloc((size_t) d.n * d.p, sizeof(double));
  for (ptrdiff_t t = 0; t < d.n; t++) {
    for (int i = 0; i < d.p; i++) {
      yd[(t * d.p) + i] = y[(t * d.p) + i] - dv[(t * d.dd) + i];
    }
  }

  return yd;
}

transformed_model transform_model(model_dims d, double *y, double *Z,
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
    double *cvec,
    int dZ,
    int dH,
    int dT,
    int dR,
    int dQ,
    int dc,
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
    double *Pinf,
    double *Finf,
    double *Minf,
    int *d_end,
    int *j_end,
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
  double *Pi   = (double *) R_alloc(m * m, sizeof(double));
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

  // Exact diffuse initialisation (5.2, univariate form 6.4): P_1 = P_* +
  // kappa P_inf with kappa -> infinity. While P_inf is non-zero, P_inf
  // (stored at Pinf, one slot per step up to the end of the diffuse phase)
  // is propagated alongside P = P_*. Zero tests for F_inf and P_inf are
  // relative to the scale of P_inf at the start.
  bool diffuse = false;
  double pinf_ref = 0.0;
  if (Pinf != NULL) {
    for (int k = 0; k < m * m; k++) {
      if (fabs(Pinf[k]) > pinf_ref) pinf_ref = fabs(Pinf[k]);
    }
    diffuse = pinf_ref > 0.0;
  }
  *d_end = 0;
  *j_end = 0;
  
  // Filter
  for (ptrdiff_t t = 0; t < n; t++) {
    double *Z_t = Z + (t * dZ);
    double *H_t = H + (t * dH);
    double *T_t = T + (t * dT);

    memcpy(att + (t * s_m), a + (t * s_m), m * sizeof(double));
    memcpy(Ptt + (t * s_mm), P + (t * s_mm), m * m * sizeof(double));
    if (diffuse) memcpy(Pi, Pinf + (t * s_mm), m * m * sizeof(double));
    
    for (int i = 0; i < p; i++) {
      // A missing observation carries no information: no update, and
      // Finv = 0 makes the smoother pass r and N through unchanged
      if (ISNAN(y[(t * p) + i])) {
        v[(t * s_p) + i] = NA_REAL;
        F[(t * s_p) + i] = NA_REAL;
        Finv[(t * s_p) + i] = 0.0;
        fill_na(K + (t * s_mp) + (i * m), m);
        if (Finf != NULL) Finf[(t * s_p) + i] = 0.0;
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
      if (Finf != NULL) Finf[(t * s_p) + i] = 0.0;

      if (diffuse) {
        // M_inf = P_inf Z_i', F_inf = Z_i M_inf
        double *Mi = Minf + (t * s_mp) + (i * m);
        double *Ms = K + (t * s_mp) + (i * m);
        F77_CALL(dsymv)(
          up, &m,
          &pos, Pi, &m, Z_t + i, &p,
          &zero, Mi, &inc, len_1);
        double Fi = F77_CALL(ddot)(&m, Mi, &inc, Z_t + i, &p);
        double z2 = F77_CALL(ddot)(&m, Z_t + i, &p, Z_t + i, &p);

        if (Fi > sqrt(DBL_EPSILON) * pinf_ref * z2) {
          // Diffuse update (5.12), with K_inf = M_inf / F_inf:
          //   a     += K_inf v
          //   P_*   += K_inf K_inf' F_* - K_inf M_*' - M_* K_inf'
          //   P_inf -= K_inf M_inf'
          double Fs = F[(t * s_p) + i];
          Finf[(t * s_p) + i] = Fi;
          Finv[(t * s_p) + i] = 0.0;
          c = v[(t * s_p) + i] / Fi;
          F77_CALL(daxpy)(&m, &c, Mi, &inc, att + (t * s_m), &inc);
          c = Fs / (Fi * Fi);
          F77_CALL(dger)(&m, &m, &c, Mi, &inc, Mi, &inc, Ptt + (t * s_mm), &m);
          c = -1.0 / Fi;
          F77_CALL(dger)(&m, &m, &c, Mi, &inc, Ms, &inc, Ptt + (t * s_mm), &m);
          F77_CALL(dger)(&m, &m, &c, Ms, &inc, Mi, &inc, Ptt + (t * s_mm), &m);
          F77_CALL(dger)(&m, &m, &c, Mi, &inc, Mi, &inc, Pi, &m);

          // Diffuse log-likelihood term; the 2 pi constant is not counted
          // for diffuse observations (as in KFAS)
          ll_sum += log(Fi);

          *d_end = (int) t + 1;
          *j_end = i + 1;
          double pinf_max = 0.0;
          for (int k = 0; k < m * m; k++) {
            if (fabs(Pi[k]) > pinf_max) pinf_max = fabs(Pi[k]);
          }
          if (pinf_max <= sqrt(DBL_EPSILON) * pinf_ref) diffuse = false;
          continue;
        }
      }

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
    if (cvec != NULL) {
      F77_CALL(daxpy)(&m, &pos, cvec + (t * dc), &inc, a + ((t + 1) * s_m), &inc);
    }
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

    // Predict P_inf: T P_inf T' (no disturbance term)
    if (diffuse) {
      F77_CALL(dgemm)(
        nt, nt, &m, &m, &m,
        &pos, T_t, &m, Pi, &m,
        &zero, mm, &m, len_1, len_2);
      F77_CALL(dgemm)(
        nt, tr, &m, &m, &m,
        &pos, mm, &m, T_t, &m,
        &zero, Pinf + ((t + 1) * s_mm), &m, len_1, len_2);
    }
  }

  *loglik = -0.5 * ((double) n_obs * LOG_2PI + ll_sum);

  if (smoother || disturbances) {
    if (smoother) {
      // Initialise alphahat = a, and V = P
      memcpy(alphahat, a, (size_t) n * m * sizeof(double));
      memcpy(V, P, (size_t) n * m * m * sizeof(double));
    }

    // Diffuse smoother (5.3): during the diffuse phase (t < d) the extra
    // recursions r1, N1 and N2 run alongside r0 = r and N0 = N. They are
    // zero after the diffuse phase, so they start at zero.
    int d = *d_end;
    double *r1 = NULL;
    double *r1n = NULL;
    double *Ki = NULL;
    double *K0 = NULL;
    double *N1 = NULL;
    double *N2 = NULL;
    double *N1n = NULL;
    double *N2n = NULL;
    double *Linf = NULL;
    double *L0 = NULL;
    double *tmp = NULL;
    if (d > 0) {
      r1   = (double *) S_alloc(m,     sizeof(double));
      r1n  = (double *) R_alloc(m,     sizeof(double));
      Ki   = (double *) R_alloc(m,     sizeof(double));
      K0   = (double *) R_alloc(m,     sizeof(double));
      N1   = (double *) S_alloc(m * m, sizeof(double));
      N2   = (double *) S_alloc(m * m, sizeof(double));
      N1n  = (double *) R_alloc(m * m, sizeof(double));
      N2n  = (double *) R_alloc(m * m, sizeof(double));
      Linf = (double *) R_alloc(m * m, sizeof(double));
      L0   = (double *) R_alloc(m * m, sizeof(double));
      tmp  = (double *) R_alloc(m * m, sizeof(double));
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

      bool in_diffuse = t < d;
      if (in_diffuse) {
        // r1 = T' r1, N1 = T' N1 T, N2 = T' N2 T
        F77_CALL(dgemv)(tr, &m, &m, &pos, T_t, &m, r1, &inc, &zero, r1n, &inc, len_1);
        memcpy(r1, r1n, m * sizeof(double));
        F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N1, &m, T_t, &m, &zero, tmp, &m, len_1, len_2);
        F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, T_t, &m, tmp, &m, &zero, N1, &m, len_1, len_2);
        F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N2, &m, T_t, &m, &zero, tmp, &m, len_1, len_2);
        F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, T_t, &m, tmp, &m, &zero, N2, &m, len_1, len_2);
      }

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

        if (in_diffuse && Finf[(t * p) + i] > 0.0) {
          // Diffuse element (5.21-5.23, univariate), with z = Z_i',
          // K_inf = M_inf / F_inf, K_0 = (M_* - K_inf F_*) / F_inf,
          // L_inf = I - K_inf z' and L_0 = -K_0 z'
          double Fi = Finf[(t * p) + i];
          double Fs = F[(t * p) + i];
          double *Mi = Minf + (t * m * p) + (i * m);
          double *Ms = K + (t * m * p) + (i * m);
          double *r0 = r + (t * m);
          double *N0 = N + (t * m * m);
          for (int k = 0; k < m; k++) {
            Ki[k] = Mi[k] / Fi;
            K0[k] = (Ms[k] - Ki[k] * Fs) / Fi;
          }

          if (disturbances) {
            // epshat = -H K_inf' r0, Var(eps | y) = H - H^2 K_inf' N0 K_inf
            F77_CALL(dsymv)(up, &m, &pos, N0, &m, Ki, &inc, &zero, m1, &inc, len_1);
            epshat[(t * p) + i] =
              -H_t[(i * p) + i] * F77_CALL(ddot)(&m, Ki, &inc, r0, &inc);
            V_eps[(t * p) + i] = H_t[(i * p) + i] - H_t[(i * p) + i] *
              H_t[(i * p) + i] * F77_CALL(ddot)(&m, Ki, &inc, m1, &inc);
          }

          memset(Linf, 0, m * m * sizeof(double));
          for (int k = 0; k < m; k++) Linf[(k * m) + k] = 1.0;
          F77_CALL(dger)(&m, &m, &neg, Ki, &inc, Z_t + i, &p, Linf, &m);
          memset(L0, 0, m * m * sizeof(double));
          F77_CALL(dger)(&m, &m, &neg, K0, &inc, Z_t + i, &p, L0, &m);

          // r1 = z v / F_inf + L_0' r0 + L_inf' r1, r0 = L_inf' r0
          F77_CALL(dgemv)(tr, &m, &m, &pos, Linf, &m, r1, &inc, &zero, r1n, &inc, len_1);
          F77_CALL(dgemv)(tr, &m, &m, &pos, L0, &m, r0, &inc, &pos, r1n, &inc, len_1);
          c = v[(t * p) + i] / Fi;
          F77_CALL(daxpy)(&m, &c, Z_t + i, &p, r1n, &inc);
          memcpy(r1, r1n, m * sizeof(double));
          F77_CALL(dgemv)(tr, &m, &m, &pos, Linf, &m, r0, &inc, &zero, m1, &inc, len_1);
          memcpy(r0, m1, m * sizeof(double));

          // N2 = -z z' F_* / F_inf^2 + L_inf' N2 L_inf + L_0' N1' L_inf
          //      + L_inf' N1 L_0 + L_0' N0 L_0
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N2, &m, Linf, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, Linf, &m, tmp, &m, &zero, N2n, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, N1, &m, Linf, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, L0, &m, tmp, &m, &pos, N2n, &m, len_1, len_2);
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N1, &m, L0, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, Linf, &m, tmp, &m, &pos, N2n, &m, len_1, len_2);
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N0, &m, L0, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, L0, &m, tmp, &m, &pos, N2n, &m, len_1, len_2);
          c = -Fs / (Fi * Fi);
          F77_CALL(dger)(&m, &m, &c, Z_t + i, &p, Z_t + i, &p, N2n, &m);

          // N1 = z z' / F_inf + L_inf' N1 L_inf + L_0' N0 L_inf
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N1, &m, Linf, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, Linf, &m, tmp, &m, &zero, N1n, &m, len_1, len_2);
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N0, &m, Linf, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, L0, &m, tmp, &m, &pos, N1n, &m, len_1, len_2);
          c = 1.0 / Fi;
          F77_CALL(dger)(&m, &m, &c, Z_t + i, &p, Z_t + i, &p, N1n, &m);

          // N0 = L_inf' N0 L_inf
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, N0, &m, Linf, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(tr, nt, &m, &m, &m, &pos, Linf, &m, tmp, &m, &zero, N0, &m, len_1, len_2);

          memcpy(N1, N1n, m * m * sizeof(double));
          memcpy(N2, N2n, m * m * sizeof(double));
          continue;
        }

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

        // During the diffuse phase, N1 = N1 L with L = I - K z'
        if (in_diffuse) {
          F77_CALL(dgemv)(nt, &m, &m, &Finv_ti, N1, &m, M_ti, &inc, &zero, r1n, &inc, len_1);
          F77_CALL(dger)(&m, &m, &neg, r1n, &inc, Z_t + i, &p, N1, &m);
        }
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

        if (in_diffuse) {
          // alphahat += P_inf r1 (5.24)
          // V -= P_inf N1 P + (P_inf N1 P)' + P_inf N2 P_inf (5.29)
          double *Pinf_t = Pinf + (t * m * m);
          double *V_t = V + (t * m * m);
          F77_CALL(dsymv)(
            up, &m,
            &pos, Pinf_t, &m, r1, &inc,
            &pos, alphahat + (t * m), &inc, len_1);
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, Pinf_t, &m, N1, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, tmp, &m, P + (t * m * m), &m, &zero, mm, &m, len_1, len_2);
          for (int k = 0; k < m; k++) {
            for (int l = 0; l < m; l++) V_t[l + (k * m)] -= mm[l + (k * m)] + mm[k + (l * m)];
          }
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &pos, Pinf_t, &m, N2, &m, &zero, tmp, &m, len_1, len_2);
          F77_CALL(dgemm)(nt, nt, &m, &m, &m, &neg, tmp, &m, Pinf_t, &m, &pos, V_t, &m, len_1, len_2);
        }
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
    SEXP P1inf,
    SEXP dvec,
    SEXP cvec,
    SEXP smoother,
    SEXP disturbances
) {
  model_dims d = check_model(y, Z, H, T, R, Q, a1, P1, P1inf, dvec, cvec);
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

  int list_length = 20;

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

  transformed_model tm = transform_model(
    d, subtract_intercept(d, y_ptr, dvec), Z_ptr, H_ptr);
  bool diagonal = tm.diagonal;
  double *L = tm.L;
  int dL = tm.dL;
  int *n_obs = tm.n_obs;
  int *obs = tm.obs;

  // Original-scale V_eps for non-diagonal H needs the smoothed V
  bool run_smoother = do_smoother || (do_disturbances && !diagonal);

  // Diffuse initialisation: P_inf for every step (slot 0 is P1inf), F_inf
  // and M_inf, kept for the smoother
  double *Pinf_buf = NULL;
  double *Finf_buf = NULL;
  double *Minf_buf = NULL;
  int d_end = 0;
  int j_end = 0;
  if (any_nonzero(REAL(P1inf), (R_xlen_t) m * m)) {
    Pinf_buf = (double *) R_alloc((size_t) (n + 1) * m * m, sizeof(double));
    Finf_buf = (double *) R_alloc((size_t) n * p, sizeof(double));
    Minf_buf = (double *) R_alloc((size_t) n * m * p, sizeof(double));
    memcpy(Pinf_buf, REAL(P1inf), m * m * sizeof(double));
  }

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
    isNull(cvec) ? NULL : REAL(cvec),
    tm.dZ,
    tm.dH,
    d.dT,
    d.dR,
    d.dQ,
    d.dc,
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
    Pinf_buf,
    Finf_buf,
    Minf_buf,
    &d_end,
    &j_end,
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

  // P_inf and F_inf over the diffuse phase (time steps 1..d), as in KFAS
  SEXP Pinf_out = PROTECT(alloc3DArray(REALSXP, m, m, d_end));
  SEXP Finf_out = PROTECT(allocMatrix(REALSXP, p, d_end));
  if (d_end > 0) {
    memcpy(REAL(Pinf_out), Pinf_buf, (size_t) d_end * m * m * sizeof(double));
    memcpy(REAL(Finf_out), Finf_buf, (size_t) d_end * p * sizeof(double));
  }
  SET_VECTOR_ELT(result, 16, Pinf_out);
  SET_VECTOR_ELT(result, 17, Finf_out);
  SET_VECTOR_ELT(result, 18, ScalarInteger(d_end));
  SET_VECTOR_ELT(result, 19, ScalarInteger(j_end));

  UNPROTECT(19);

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
    SEXP P1,
    SEXP P1inf,
    SEXP dvec,
    SEXP cvec
) {
  model_dims d = check_model(y, Z, H, T, R, Q, a1, P1, P1inf, dvec, cvec);
  int p = d.p;
  int m = d.m;

  transformed_model tm = transform_model(
    d, subtract_intercept(d, REAL(y), dvec), REAL(Z), REAL(H));

  // Single-step filter buffers
  double *v   = (double *) R_alloc(p,     sizeof(double));
  double *F   = (double *) R_alloc(p,     sizeof(double));
  double *att = (double *) R_alloc(m,     sizeof(double));
  double *Ptt = (double *) R_alloc(m * m, sizeof(double));
  double *a   = (double *) R_alloc(m,     sizeof(double));
  double *P   = (double *) R_alloc(m * m, sizeof(double));
  double *K   = (double *) R_alloc(m * p, sizeof(double));
  double loglik;
  int d_end;
  int j_end;

  // Single-step diffuse buffers
  double *Pinf = NULL;
  double *Finf = NULL;
  double *Minf = NULL;
  if (any_nonzero(REAL(P1inf), (R_xlen_t) m * m)) {
    Pinf = (double *) R_alloc(m * m, sizeof(double));
    Finf = (double *) R_alloc(p,     sizeof(double));
    Minf = (double *) R_alloc(m * p, sizeof(double));
    memcpy(Pinf, REAL(P1inf), m * m * sizeof(double));
  }

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
    isNull(cvec) ? NULL : REAL(cvec),
    tm.dZ,
    tm.dH,
    d.dT,
    d.dR,
    d.dQ,
    d.dc,
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
    Pinf,
    Finf,
    Minf,
    &d_end,
    &j_end,
    false,
    false,
    false
  );

  return ScalarReal(loglik - log_det_L(tm, d.n));
}

