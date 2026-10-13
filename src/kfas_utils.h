// Helpers shared by the univariate (kfas0) and multivariate (kfas) routines
// Byron Botha 2025

#ifndef KFAS_UTILS_H
#define KFAS_UTILS_H

#include <Rinternals.h>

#define LOG_2PI 1.837877066409345483560659472811  // log(2 pi)

// State disturbance variance RQR' (mq is m x q scratch)
void compute_RQR(int m, int q, double *R, double *Q, double *mq,
                 double *RQR);

// QR', used by the eta disturbance smoother
void compute_QR(int m, int q, double *R, double *Q, double *QR);

// Offset between time steps of a system matrix with `size` elements per
// step: 0 if time-invariant, `size` if time-varying (n slices). Errors if
// the length matches neither.
int stride(SEXP x, int size, int n, const char *name);

// Error unless x is a double vector of the given length (size < 0: any)
void check_double(SEXP x, R_xlen_t size, const char *name);

// Error unless x is a double matrix or array (at least two dimensions)
void check_matrix(SEXP x, const char *name);

void fill_na(double *x, R_xlen_t len);

#endif
