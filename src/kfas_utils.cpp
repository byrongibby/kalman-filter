// Helpers shared by the univariate (kfas0) and multivariate (kfas) routines
// Byron Botha 2025

#include <stddef.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "kfas_utils.h"

// State disturbance variance RQR' (mq is m x q scratch)
void compute_RQR(int m, int q, double *R, double *Q, double *mq,
                 double *RQR) {
  const char *tr = "t";
  const char *nt = "n";
  double pos = 1.0;
  double zero = 0.0;
  FC_LEN_T len_1 = 1;
  FC_LEN_T len_2 = 1;

  F77_CALL(dgemm)(
    nt, nt, &m, &q, &q,
    &pos, R, &m, Q, &q,
    &zero, mq, &m, len_1, len_2);
  F77_CALL(dgemm)(
    nt, tr, &m, &m, &q,
    &pos, mq, &m, R, &m,
    &zero, RQR, &m, len_1, len_2);
}

// QR', used by the eta disturbance smoother
void compute_QR(int m, int q, double *R, double *Q, double *QR) {
  const char *tr = "t";
  const char *nt = "n";
  double pos = 1.0;
  double zero = 0.0;
  FC_LEN_T len_1 = 1;
  FC_LEN_T len_2 = 1;

  F77_CALL(dgemm)(
    nt, tr, &q, &m, &q,
    &pos, Q, &q, R, &m,
    &zero, QR, &q, len_1, len_2);
}

// System matrices may be time-invariant (one matrix) or time-varying (n
// matrices stacked in an array). The stride is the offset between time
// steps: 0 for time-invariant, so X + t * dX works in both cases.
int stride(SEXP x, int size, int n, const char *name) {
  R_xlen_t len = XLENGTH(x);

  if (len == size) return 0;
  if (len == (R_xlen_t) size * n) return size;
  error("%s must have %d (time-invariant) or %lld (time-varying) "
        "elements, not %lld", name, size, (long long) size * n,
        (long long) len);
}

// Error unless x is a double vector of the given length
void check_double(SEXP x, R_xlen_t size, const char *name) {
  if (TYPEOF(x) != REALSXP) error("%s must be a double vector or array", name);
  if (size >= 0 && XLENGTH(x) != size) {
    error("%s must have %lld elements, not %lld", name, (long long) size,
          (long long) XLENGTH(x));
  }
}

// Error unless x is a double matrix or array (at least two dimensions)
void check_matrix(SEXP x, const char *name) {
  check_double(x, -1, name);
  if (length(getAttrib(x, R_DimSymbol)) < 2) {
    error("%s must be a matrix or array", name);
  }
}

void fill_na(double *x, R_xlen_t len) {
  for (R_xlen_t i = 0; i < len; i++) x[i] = NA_REAL;
}
