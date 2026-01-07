#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "pcg_basic.h"

// Box-Muller transformation.
static void rnorm(pcg32_random_t* rng, double *z, int n) {
  assert(rng);
  assert(z);
  assert(n > 0);

  int i;
  double u1, u2;

  // Floating point values in the range [0,1) that have been
  // rounded down to the nearest multiple of 1/2^32.
  u1 = ldexp(pcg32_random_r(rng), -32);
  u2 = ldexp(pcg32_random_r(rng), -32);

  for (i = 0; i < n - 2; i += 2) {
    z[i]     = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    z[i + 1] = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

    u1 = ldexp(pcg32_random_r(rng), -32);
    u2 = ldexp(pcg32_random_r(rng), -32);
  }

  z[i - 1] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
  if (n - i == 2) z[i] = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
}

int mvrnorm_c(double *x, int n, int d, double *mu, double *sigma) {
  assert(n > 0);
  assert(d > 0);
  assert(mu);
  assert(sigma);

  // Initialise PCG random number generator
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, time(NULL), (intptr_t)&rng);

  // Vars required for BLAS/LAPACK routines
  char *nt = "n";
  char *lo = "l";
  int info;
  double pos = 1.0;
  
  // String lengths for cstring args
  FC_LEN_T len_1 = 1; 
  FC_LEN_T len_2 = 1;

  double *z = malloc(d * n * sizeof(double));
  double *C = malloc(d * d * sizeof(double));

  rnorm(&rng, z, d * n);

  for (int i = 0; i < n; i++) memcpy(x + i * d, mu, d * sizeof(double));
  memcpy(C, sigma, d * d * sizeof(double));

  F77_CALL(dpotrf)(
    lo, &d, C, &d, &info, len_1);

  for (int j = 0; j < d; j++) {
    for (int i = 0; i < j; i++) {
        C[i + j * d] = 0.0;
    }
  }

  F77_CALL(dgemm)(
    nt, nt, &d, &n, &d,
    &pos, C, &d, z, &d,
    &pos, x, &d, len_1, len_2);

  free(z);
  free(C);

  return info;
}

SEXP mvrnorm(SEXP n, SEXP mu, SEXP sigma) {
  SEXP dim_sigma = PROTECT(getAttrib(sigma, R_DimSymbol));
  
  int len = INTEGER(n)[0];
  int dim = INTEGER(dim_sigma)[0];

  SEXP x = PROTECT(allocMatrix(REALSXP, dim, len));

  double *mu_ptr    = REAL(mu);
  double *sigma_ptr = REAL(sigma);
  double *x_ptr     = REAL(x);

  if (mvrnorm_c(x_ptr, len, dim, mu_ptr, sigma_ptr) != 0) {
    printf("Sample draw failed!\n");
  }

  UNPROTECT(2);

  return x;
}
