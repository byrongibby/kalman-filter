// Model checks and transformations shared by the univariate filter (kfas0)
// and the precision sampler
// Byron Botha 2025

#ifndef KFAS_MODEL_H
#define KFAS_MODEL_H

#include <stdbool.h>
#include <Rinternals.h>

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
  int dd;     // d_t and c_t (intercepts), if given
  int dc;
} model_dims;

model_dims check_model(SEXP y, SEXP Z, SEXP H, SEXP T, SEXP R, SEXP Q,
                       SEXP a1, SEXP P1, SEXP P1inf, SEXP dvec, SEXP cvec);

bool any_nonzero(double *x, R_xlen_t len);

// Observation intercept: the model is applied to y - d, so v = y - d - Z a.
// Returns y itself if there is no intercept.
double *subtract_intercept(model_dims d, double *y, SEXP dvec);

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

transformed_model transform_model(model_dims d, double *y, double *Z,
                                  double *H);

#endif
