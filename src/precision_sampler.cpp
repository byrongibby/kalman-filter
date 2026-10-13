// Precision-based sampler for the states (Chan and Jeliazkov, 2009)
// Byron Botha 2025

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "kfas_utils.h"
#include "kfas_model.h"

// B, times scale, added to the m x m block at block row bi and block column
// bj (bi >= bj) of a symmetric matrix in LAPACK lower band storage with
// leading dimension ldab. Only elements on or below the diagonal are stored,
// so only the lower triangle of a diagonal block (bi == bj) is read.
static void add_block(double *AB, int ldab, int m, ptrdiff_t bi,
                      ptrdiff_t bj, double *B, double scale) {
  for (int c = 0; c < m; c++) {
    ptrdiff_t col = (bj * m) + c;
    for (int r = 0; r < m; r++) {
      ptrdiff_t row = (bi * m) + r;
      if (row >= col) AB[(row - col) + (col * ldab)] += scale * B[r + (c * m)];
    }
  }
}

// Lower Cholesky factor of a symmetric positive definite m x m matrix, in
// place. Returns false if A is not positive definite, including when it is
// singular but rounding leaves a tiny positive pivot: each squared pivot (a
// conditional variance) must exceed rounding error relative to the largest
// diagonal element.
static bool spd_cholesky(double *A, int m) {
  const char *lo = "l";
  int info;
  FC_LEN_T len_1 = 1;

  double scale = 0.0;
  for (int j = 0; j < m; j++) {
    if (A[(j * m) + j] > scale) scale = A[(j * m) + j];
  }
  F77_CALL(dpotrf)(lo, &m, A, &m, &info, len_1);
  if (info != 0) return false;
  for (int j = 0; j < m; j++) {
    double pivot = A[(j * m) + j];
    if (pivot * pivot <= 100.0 * m * DBL_EPSILON * scale) return false;
  }

  return true;
}

// Inverse of a symmetric positive definite m x m matrix, in place (both
// triangles). Returns false if A is not positive definite (see
// spd_cholesky).
static bool spd_inverse(double *A, int m) {
  const char *lo = "l";
  int info;
  FC_LEN_T len_1 = 1;

  if (!spd_cholesky(A, m)) return false;
  F77_CALL(dpotri)(lo, &m, A, &m, &info, len_1);
  if (info != 0) return false;
  for (int c = 0; c < m; c++) {
    for (int r = 0; r < c; r++) A[r + (c * m)] = A[c + (r * m)];
  }

  return true;
}

// x += Dt' (r - A x), with A and Dt k x N; e (k) is scratch
static void constrain(int N, int k, double *A, double *Dt, double *r,
                      double *e, double *x) {
  const char *tr = "t";
  const char *nt = "n";
  int inc = 1;
  double pos = 1.0;
  double neg = -1.0;
  FC_LEN_T len_1 = 1;

  memcpy(e, r, k * sizeof(double));
  F77_CALL(dgemv)(nt, &k, &N, &neg, A, &k, x, &inc, &pos, e, &inc, len_1);
  F77_CALL(dgemv)(tr, &k, &N, &pos, Dt, &k, e, &inc, &pos, x, &inc, len_1);
}

// Prior precision of alpha_1 ~ N(a1, P1 + kappa P1inf) as kappa -> infinity:
// P1^-1 without diffuse elements, otherwise B (B' P1 B)^-1 B' with the
// columns of B spanning the null space of P1inf (a flat prior on its column
// space)
static void initial_precision(int m, double *P1, double *P1inf,
                              double *prec) {
  const char *tr = "t";
  const char *nt = "n";
  const char *lo = "l";
  const char *jv = "v";
  int info;
  double pos = 1.0;
  double zero = 0.0;
  FC_LEN_T len_1 = 1;
  FC_LEN_T len_2 = 1;

  if (P1inf == NULL) {
    memcpy(prec, P1, m * m * sizeof(double));
    if (!spd_inverse(prec, m)) {
      error("P1 must be positive definite for the precision sampler, "
            "apart from diffuse elements (P1inf)");
    }
    return;
  }

  // Eigenvectors of P1inf, eigenvalues in ascending order: the first k
  // (numerically zero) span its null space
  double *B    = (double *) R_alloc(m * m, sizeof(double));
  double *w    = (double *) R_alloc(m,     sizeof(double));
  int lwork = 3 * m;
  double *work = (double *) R_alloc(lwork, sizeof(double));
  memcpy(B, P1inf, m * m * sizeof(double));
  F77_CALL(dsyev)(jv, lo, &m, B, &m, w, work, &lwork, &info, len_1, len_2);
  if (info != 0) error("eigendecomposition of P1inf failed");

  int k = 0;
  while (k < m && w[k] <= sqrt(DBL_EPSILON) * w[m - 1]) k++;
  if (k == 0) {
    memset(prec, 0, m * m * sizeof(double));
    return;
  }

  // C = (B' P1 B)^-1, prec = B C B'
  double *PB = (double *) R_alloc(m * k, sizeof(double));
  double *C  = (double *) R_alloc(k * k, sizeof(double));
  F77_CALL(dgemm)(
    nt, nt, &m, &k, &m,
    &pos, P1, &m, B, &m,
    &zero, PB, &m, len_1, len_2);
  F77_CALL(dgemm)(
    tr, nt, &k, &k, &m,
    &pos, B, &m, PB, &m,
    &zero, C, &k, len_1, len_2);
  if (!spd_inverse(C, k)) {
    error("P1 must be positive definite for the precision sampler, "
          "apart from diffuse elements (P1inf)");
  }
  F77_CALL(dgemm)(
    nt, nt, &m, &k, &k,
    &pos, B, &m, C, &k,
    &zero, PB, &m, len_1, len_2);
  F77_CALL(dgemm)(
    nt, tr, &m, &m, &k,
    &pos, PB, &m, B, &m,
    &zero, prec, &m, len_1, len_2);
}

// Draws from p(alpha_1, ..., alpha_n | y). Stacking the states as
// alpha = (alpha_1', ..., alpha_n')', the transition equation is
// G alpha = b + u with G block lower bidiagonal (identity blocks on the
// diagonal, -T_t below it), b = (a1', c_1', ..., c_n-1')' and
// u ~ N(0, S), S = diag(P1, W_1, ..., W_n-1), W_t = R_t Q_t R_t'. The
// posterior precision and the linear term of its mean are
//   K = G' S^-1 G + diag(Z_t' H_t^-1 Z_t)
//   K mu = G' S^-1 b + (Z_t' H_t^-1 (y_t - d_t))_t
// K is block tridiagonal, a band matrix with bandwidth 2m - 1, so
// K = L L' (banded Cholesky) costs O(n m^3), mu follows from two banded
// triangular solves, and alpha = mu + L'^-1 z with z ~ N(0, I) is a draw.
// Missing observations drop out of the sum; non-diagonal H uses the
// transformed model with H* = I.
//
// Optional linear constraints A alpha = r (k x N, k < N) truncate the
// posterior to a hyperplane, using algorithm 2 of Cong, Chen and Zhou
// (2017): with B = K^-1 A' and C = A B = A K^-1 A', each unconstrained
// draw x becomes x + B C^-1 (r - A x), an exact draw from the posterior
// conditional on A alpha = r, and the mean becomes the conditional mean.
// B costs k banded solves, once; each draw then costs O(N k).
extern "C" SEXP precision_sampler(
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
    SEXP nsim,
    SEXP Amat,
    SEXP rvec
) {
  model_dims d = check_model(y, Z, H, T, R, Q, a1, P1, P1inf, dvec, cvec);
  int n = d.n;
  int p = d.p;
  int m = d.m;
  int q = d.q;
  int dT = d.dT;
  int dR = d.dR;
  int dQ = d.dQ;
  int dc = d.dc;

  int ns = asInteger(nsim);
  if (ns == NA_INTEGER || ns < 1) error("nsim must be a positive integer");
  if ((double) n * m > INT_MAX) error("n * m is too large");

  // Constraints A alpha = r, if given: A is k x N with the columns in the
  // order of the stacked states (state fastest)
  int k = 0;
  if (!isNull(Amat)) {
    check_double(rvec, -1, "rhs");
    if (XLENGTH(rvec) < 1 || XLENGTH(rvec) >= (R_xlen_t) n * m) {
      error("there must be between 1 and n * m - 1 constraints");
    }
    k = (int) XLENGTH(rvec);
    check_double(Amat, (R_xlen_t) k * n * m, "constraints");
  }

  // Scalars required for BLAS/LAPACK routines
  const char *tr = "t";
  const char *nt = "n";
  const char *lo = "l";
  const char *nu = "n";
  int inc = 1;
  double pos = 1.0;
  double neg = -1.0;
  double zero = 0.0;
  int info;
  FC_LEN_T len_1 = 1;
  FC_LEN_T len_2 = 1;

  transformed_model tm = transform_model(
    d, subtract_intercept(d, REAL(y), dvec), REAL(Z), REAL(H));
  double *T_ptr = REAL(T);
  double *R_ptr = REAL(R);
  double *Q_ptr = REAL(Q);
  double *c_ptr = isNull(cvec) ? NULL : REAL(cvec);

  // K in lower band storage, and the linear term K mu
  int N = n * m;
  int kd = (2 * m) - 1;
  if (kd > N - 1) kd = N - 1;
  int ldab = kd + 1;
  double *AB  = (double *) S_alloc((size_t) ldab * N, sizeof(double));
  double *lin = (double *) S_alloc(N, sizeof(double));

  double *prec = (double *) R_alloc(m * m, sizeof(double));
  double *W    = (double *) R_alloc(m * m, sizeof(double));
  double *TW   = (double *) R_alloc(m * m, sizeof(double));
  double *mm   = (double *) R_alloc(m * m, sizeof(double));
  double *mq   = (double *) R_alloc(m * q, sizeof(double));
  double *m1   = (double *) R_alloc(m,     sizeof(double));
  double *Zc   = (double *) R_alloc(p * m, sizeof(double));
  double *yc   = (double *) R_alloc(p,     sizeof(double));

  // Initial state: precision P1^-1 (or its diffuse limit), linear term
  // P1^-1 a1
  bool diffuse = any_nonzero(REAL(P1inf), (R_xlen_t) m * m);
  initial_precision(m, REAL(P1), diffuse ? REAL(P1inf) : NULL, prec);
  add_block(AB, ldab, m, 0, 0, prec, 1.0);
  F77_CALL(dgemv)(
    nt, &m, &m,
    &pos, prec, &m, REAL(a1), &inc,
    &pos, lin, &inc, len_1);

  // Transitions alpha_t+1 = c_t + T_t alpha_t + u_t, u_t ~ N(0, W_t):
  //   K_t,t     += T_t' W_t^-1 T_t
  //   K_t+1,t+1 += W_t^-1
  //   K_t+1,t   -= W_t^-1 T_t
  //   (K mu)_t+1 += W_t^-1 c_t,  (K mu)_t -= T_t' W_t^-1 c_t
  bool varying_W = (dR != 0 || dQ != 0);
  for (ptrdiff_t t = 0; t < n - 1; t++) {
    double *T_t = T_ptr + (t * dT);

    if (t == 0 || varying_W) {
      compute_RQR(m, q, R_ptr + (t * dR), Q_ptr + (t * dQ), mq, W);
      if (!spd_inverse(W, m)) {
        error("R Q R' is not positive definite at time %lld: the precision "
              "sampler needs a non-singular state disturbance variance",
              (long long) t + 1);
      }
    }

    // TW = T' W^-1
    F77_CALL(dgemm)(
      tr, nt, &m, &m, &m,
      &pos, T_t, &m, W, &m,
      &zero, TW, &m, len_1, len_2);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, TW, &m, T_t, &m,
      &zero, mm, &m, len_1, len_2);
    add_block(AB, ldab, m, t, t, mm, 1.0);
    add_block(AB, ldab, m, t + 1, t + 1, W, 1.0);
    F77_CALL(dgemm)(
      nt, nt, &m, &m, &m,
      &pos, W, &m, T_t, &m,
      &zero, mm, &m, len_1, len_2);
    add_block(AB, ldab, m, t + 1, t, mm, -1.0);

    if (c_ptr != NULL) {
      F77_CALL(dgemv)(
        nt, &m, &m,
        &pos, W, &m, c_ptr + (t * dc), &inc,
        &zero, m1, &inc, len_1);
      F77_CALL(daxpy)(&m, &pos, m1, &inc, lin + ((t + 1) * m), &inc);
      F77_CALL(dgemv)(
        tr, &m, &m,
        &neg, T_t, &m, m1, &inc,
        &pos, lin + (t * m), &inc, len_1);
    }
  }

  // Observations, each independent given the states (H diagonal, or H* = I
  // after the transform): with the observed rows of Z_t and y_t scaled by
  // H_ii^-1/2 into Zc and yc,
  //   K_t,t += Zc' Zc,  (K mu)_t += Zc' yc
  for (ptrdiff_t t = 0; t < n; t++) {
    double *Z_t = tm.Z + (t * tm.dZ);
    double *H_t = tm.H + (t * tm.dH);
    double *y_t = tm.y + (t * p);
    int po = 0;

    for (int i = 0; i < p; i++) {
      if (ISNAN(y_t[i])) continue;
      double h = H_t[(i * p) + i];
      if (!(h > 0.0)) {
        error("H must be positive definite for the precision sampler "
              "(time %lld)", (long long) t + 1);
      }
      double s = 1.0 / sqrt(h);
      for (int j = 0; j < m; j++) Zc[po + (j * p)] = Z_t[i + (j * p)] * s;
      yc[po] = y_t[i] * s;
      po++;
    }
    if (po == 0) continue;

    F77_CALL(dsyrk)(
      lo, tr, &m, &po,
      &pos, Zc, &p,
      &zero, mm, &m, len_1, len_2);
    add_block(AB, ldab, m, t, t, mm, 1.0);
    F77_CALL(dgemv)(
      tr, &po, &m,
      &pos, Zc, &p, yc, &inc,
      &pos, lin + (t * m), &inc, len_1);
  }

  // K = L L'
  F77_CALL(dpbtrf)(lo, &N, &kd, AB, &ldab, &info, len_1);
  if (info != 0) {
    error("the posterior precision of the states is not positive definite: "
          "the states are not identified by the prior and the data");
  }

  SEXP mean   = PROTECT(allocMatrix(REALSXP, m, n));
  SEXP draws  = PROTECT(alloc3DArray(REALSXP, m, n, ns));
  SEXP result = PROTECT(allocVector(VECSXP, 2));
  double *mean_ptr  = REAL(mean);
  double *draws_ptr = REAL(draws);

  // Posterior mean: K mu = lin
  memcpy(mean_ptr, lin, (size_t) N * sizeof(double));
  F77_CALL(dpbtrs)(lo, &N, &kd, &inc, AB, &ldab, mean_ptr, &N, &info, len_1);

  // Hyperplane truncation: Dt = C^-1 B' (k x N), so that the correction
  // of x is Dt' (r - A x)
  double *A_ptr = NULL;
  double *r_ptr = NULL;
  double *Dt = NULL;
  double *e = NULL;
  if (k > 0) {
    A_ptr = REAL(Amat);
    r_ptr = REAL(rvec);
    double *B = (double *) R_alloc((size_t) N * k, sizeof(double));
    double *C = (double *) R_alloc((size_t) k * k, sizeof(double));
    Dt = (double *) R_alloc((size_t) k * N, sizeof(double));
    e  = (double *) R_alloc(k, sizeof(double));

    // B = K^-1 A'
    for (int j = 0; j < k; j++) {
      for (ptrdiff_t c = 0; c < N; c++) B[c + (j * N)] = A_ptr[j + (c * k)];
    }
    F77_CALL(dpbtrs)(lo, &N, &kd, &k, AB, &ldab, B, &N, &info, len_1);

    // C = A B, factored
    F77_CALL(dgemm)(
      nt, nt, &k, &k, &N,
      &pos, A_ptr, &k, B, &N,
      &zero, C, &k, len_1, len_2);
    if (!spd_cholesky(C, k)) {
      error("the constraints must be linearly independent");
    }

    // Dt = C^-1 B'
    for (int j = 0; j < k; j++) {
      for (ptrdiff_t c = 0; c < N; c++) Dt[j + (c * k)] = B[c + (j * N)];
    }
    F77_CALL(dpotrs)(lo, &k, &N, C, &k, Dt, &k, &info, len_1);

    constrain(N, k, A_ptr, Dt, r_ptr, e, mean_ptr);
  }

  // Draws: alpha = mu + L'^-1 z, so Var(alpha) = (L L')^-1 = K^-1, then
  // projected onto the constraints
  rnorm_seeded(draws_ptr, (ptrdiff_t) N * ns);
  for (ptrdiff_t s = 0; s < ns; s++) {
    double *x = draws_ptr + (s * N);
    F77_CALL(dtbsv)(
      lo, tr, nu, &N, &kd,
      AB, &ldab, x, &inc, len_1, len_1, len_1);
    F77_CALL(daxpy)(&N, &pos, mean_ptr, &inc, x, &inc);
    if (k > 0) constrain(N, k, A_ptr, Dt, r_ptr, e, x);
  }

  SET_VECTOR_ELT(result, 0, draws);
  SET_VECTOR_ELT(result, 1, mean);
  UNPROTECT(3);

  return result;
}
