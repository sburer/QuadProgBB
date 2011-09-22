#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

void onedim(const double c, const double sig, const double L, const double U, double *val, double *x_);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  size_t i, j, n;

  double  locval;

  double *M;
  double *L;
  double *U;
  double  sig;
  double *B;  /* Matlab will pass B_ = prhs[4] in as double, I think. */
  double *E; 

  double  val;
  double *Y;

  const mxArray *M_;
  const mxArray *L_;
  const mxArray *U_;
  const mxArray *sig_;
  const mxArray *B_;
  const mxArray *E_;

  const mxArray *val_;
  const mxArray *Y_;

  if( nrhs != 6 )
    mexErrMsgTxt("quadprob: Need six input arguments.");

  if( nlhs != 2 )
    mexErrMsgTxt("quadprob: Need two output arguments.");

  M_   = prhs[0];
  L_   = prhs[1];
  U_   = prhs[2];
  sig_ = prhs[3];
  B_   = prhs[4];
  E_   = prhs[5];

  n = mxGetM(M_) - 1;

  plhs[0] = mxCreateDoubleMatrix(1  , 1  , mxREAL);
  plhs[1] = mxCreateDoubleMatrix(1+n, 1+n, mxREAL);

  val_   = plhs[0];
  Y_     = plhs[1];

  if( mxGetM(M_) != 1+n || mxGetN(M_) != 1+n)
    mexErrMsgTxt("quadprob: Input 1 must be a square matrix of the appropriate size.");
  if( mxIsSparse(M_) )
    mexErrMsgTxt("quadprob: Input 1 should not be sparse.");
  if( mxIsDouble(M_) == 0 || mxIsComplex(M_) == 1 )
    mexErrMsgTxt("quadprob: Input 1 should be of type double.");

  if( mxGetM(L_) != 1+n || mxGetN(L_) != 1)
    mexErrMsgTxt("quadprob: Input 2 must be a column vector of the appropriate size.");
  if( mxIsSparse(L_) )
    mexErrMsgTxt("quadprob: Input 2 should not be sparse.");
  if( mxIsDouble(L_) == 0 || mxIsComplex(L_) == 1 )
    mexErrMsgTxt("quadprob: Input 2 should be of type double.");

  if( mxGetM(U_) != 1+n || mxGetN(U_) != 1)
    mexErrMsgTxt("quadprob: Input 3 must be a column vector of the appropriate size.");
  if( mxIsSparse(U_) )
    mexErrMsgTxt("quadprob: Input 3 should not be sparse.");
  if( mxIsDouble(U_) == 0 || mxIsComplex(U_) == 1 )
    mexErrMsgTxt("quadprob: Input 3 should be of type double.");

  if( mxGetM(sig_) * mxGetN(sig_) != 1)
    mexErrMsgTxt("quadprob: Input 4 must be a scalar.");
  if( mxIsSparse(sig_) )
    mexErrMsgTxt("quadprob: Input 4 should not be sparse.");
  if( mxIsDouble(sig_) == 0 || mxIsComplex(sig_) == 1 )
    mexErrMsgTxt("quadprob: Input 4 should be of type double.");

  if( mxGetM(B_) != 1+n || mxGetN(B_) != 1)
    mexErrMsgTxt("quadprob: Input 5 must be a column vector of the appropriate size.");
  if( mxIsSparse(B_) )
    mexErrMsgTxt("quadprob: Input 5 should not be sparse.");
  if( mxIsDouble(B_) == 0 || mxIsComplex(B_) == 1 )
    mexErrMsgTxt("quadprob: Input 5 should be of type double.");

  if( mxGetM(E_) != 1+n || mxGetN(E_) != 1+n)
    mexErrMsgTxt("quadprob: Input 6 must be a matrix of the appropriate size.");
  if( mxIsSparse(E_) )
    mexErrMsgTxt("quadprob: Input 6 should not be sparse.");
  if( mxIsDouble(E_) == 0 || mxIsComplex(E_) == 1 )
    mexErrMsgTxt("quadprob: Input 6 should be of type double.");

  M   =   mxGetPr(M_  ) ;
  L   =   mxGetPr(L_  ) ; 
  U   =   mxGetPr(U_  ) ; 
  sig = *(mxGetPr(sig_));
  B   =   mxGetPr(B_)   ;
  E   =   mxGetPr(E_)   ;

  Y = mxGetPr(Y_);

  if( sig < 0 )
    mexErrMsgTxt("quadprob: Input 4 should be nonegative.");


  /* L[0] = U[0] = 1.0; */ /* Code below assumes L[0] = U[0] = 1.0, which
                              is now handled in opt_dnn.m */

  /* Initialize val=0 */

  val = 0.0;

  /* Enforce Y_00=1; add to val */

  Y[0] = 1.0;
  val += M[0] + 0.5*sig;

  /* Optimize x and diag(X) portion of Y.
   * Also handle binary variables in B.
   * Add to val. */

  for(i = 1; i <= n; i++) {

    if(B[i] == 1) {

      /* This is tricky because we combine Y(i,i), Y(i,0), Y(0,i) together. */
      
      onedim(2*M[i] + M[i*(1+n)+i], 3*sig, L[i], U[i], &locval, Y+i);
      Y[i*(1+n)] = Y[i*(1+n)+i] = Y[i];
      val += locval;

    }
    else {

      onedim(M[i], sig, L[i], U[i], &locval, Y+i);
      Y[i*(1+n)] = Y[i];
      val += 2.0*locval;

      onedim(M[i*(1+n)+i], sig, L[i]*L[i], U[i]*U[i], &locval, Y+i*(1+n)+i);
      val += locval;

    }

  }

  /* Optimize strictly lower triangular part of X. */
  /* Handle complementarities. */
  /* Add to val. */

  for(j = 1; j <= n; j++) {
    for(i = j+1; i <= n; i++) {

      if(E[j*(1+n)+i] == 1.0) {
        Y[i*(1+n)+j] = Y[j*(1+n)+i] = 0.0;
      }
      else {
        onedim(M[j*(1+n)+i], sig, L[i]*L[j], U[i]*U[j], &locval, Y+j*(1+n)+i);
        Y[i*(1+n)+j] = Y[j*(1+n)+i];
        val += 2.0*locval;
      }

    }
  }

  *(mxGetPr(val_)) = val;

  return;

}

void onedim(const double c, const double sig, const double L, const double U, double *val, double *x_)
{
  double x;

  /* Solve min { c*x + 0.5*sig*x^2 : L <= x <= U } */
  /* sig may equal 0 */

  if(sig > 0.0)
    x = -c/sig;
  else
    x = L - 1.0;

  if( !(L <= x && x <= U) ) {

    if( c*L + 0.5*sig*L*L < c*U + 0.5*sig*U*U )
      x = L;
    else
      x = U;

  }

  *val = c*x + 0.5*sig*x*x;
  *x_ = x;

  return;

}
