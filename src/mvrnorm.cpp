// C++ headers must come before R headers, which define macros such as length
#include <random>
#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/Random.h>
#include "kfas_utils.h"

// Seed from R's RNG so set.seed() makes draws reproducible
static uint64_t seed_from_r(void) {
  GetRNGstate();
  uint64_t hi = (uint64_t) (unif_rand() * 4294967296.0);
  uint64_t lo = (uint64_t) (unif_rand() * 4294967296.0);
  PutRNGstate();

  return (hi << 32) | lo;
}

// Standard normal draws
static void rnorm(std::mt19937_64 *rng, double *z, ptrdiff_t n) {
  assert(rng);
  assert(z);
  assert(n > 0);

  std::normal_distribution<double> norm(0.0, 1.0);

  for (ptrdiff_t i = 0; i < n; i++) z[i] = norm(*rng);
}

static void mvrnorm_c(double *x, int n, int d, double *mu, double *sigma) {
  assert(n > 0);
  assert(d > 0);
  assert(mu);
  assert(sigma);

  // Vars required for BLAS/LAPACK routines
  const char *nt = "n";
  const char *lo = "l";
  int info;
  double pos = 1.0;

  // String lengths for cstring args
  FC_LEN_T len_1 = 1;
  FC_LEN_T len_2 = 1;

  // Cholesky factor sigma = C C', before any draws are made
  double *C = (double *) R_alloc(d * d, sizeof(double));
  memcpy(C, sigma, d * d * sizeof(double));
  F77_CALL(dpotrf)(
    lo, &d, C, &d, &info, len_1);
  if (info != 0) error("sigma is not positive definite");

  // Initialise Mersenne Twister random number generator
  std::mt19937_64 rng(seed_from_r());

  double *z = (double *) R_alloc((size_t) d * n, sizeof(double));
  rnorm(&rng, z, (ptrdiff_t) d * n);

  for (ptrdiff_t i = 0; i < n; i++) memcpy(x + i * d, mu, d * sizeof(double));

  for (int j = 0; j < d; j++) {
    for (int i = 0; i < j; i++) {
        C[i + j * d] = 0.0;
    }
  }

  F77_CALL(dgemm)(
    nt, nt, &d, &n, &d,
    &pos, C, &d, z, &d,
    &pos, x, &d, len_1, len_2);
}

extern "C" SEXP mvrnorm(SEXP n, SEXP mu, SEXP sigma) {
  int len = asInteger(n);
  if (len == NA_INTEGER || len < 1) error("n must be a positive integer");

  check_matrix(sigma, "sigma");
  SEXP dim_sigma = PROTECT(getAttrib(sigma, R_DimSymbol));
  int dim = INTEGER(dim_sigma)[0];
  if (length(dim_sigma) != 2 || INTEGER(dim_sigma)[1] != dim) {
    error("sigma must be a square matrix");
  }
  check_double(mu, dim, "mu");

  SEXP x = PROTECT(allocMatrix(REALSXP, dim, len));

  double *mu_ptr    = REAL(mu);
  double *sigma_ptr = REAL(sigma);
  double *x_ptr     = REAL(x);

  mvrnorm_c(x_ptr, len, dim, mu_ptr, sigma_ptr);

  UNPROTECT(2);

  return x;
}
