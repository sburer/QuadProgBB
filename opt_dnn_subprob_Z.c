#include <math.h>
#include "mex.h"

#if defined(_WIN32)
#define dgemm_ dgemm
#define dsymm_ dsymm
#define dsyevr_ dsyevr
#define dsyev_ dsyev
#define dsyr_ dsyr
#endif

#define MYCALLOC(VAR,TYPE,SIZE) VAR = (TYPE*)calloc(SIZE, sizeof(TYPE))
#define MYFREE(VAR) free(VAR)

#define mymin(A, B) ((A) < (B) ? (A) : (B))
#define mymax(A, B) ((A) > (B) ? (A) : (B))

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  size_t i, j;
  size_t    iOne = 1,   iZero = 0;
  double dOne = 1.0, dZero = 0.0;

  double *MAT, *subMAT, *projMAT;

  char SIDE   = 'L';
  char TRANSN = 'N';
  char TRANST = 'T';
  char JOBZ   = 'V';
  char RANGE  = 'V';
  char UPLO   = 'U';

  size_t n, k;

  double *smallMAT;
  double *TempMatrix_nk;
  double *TempMatrix_kk;

  double VL = 0, VU = 1.0e10;
  size_t    IL, IU;

  double ABSTOL = 1e-10; 

  size_t     NumEigs;
  double *EigVals;
  double *EigVecs;
  size_t    *SuppEigVals;

  double  temp_dWork;
  size_t     temp_iWork;

  double *dWork;
  size_t     dWorkLength;
  size_t    *iWork;
  size_t     iWorkLength;

  size_t INFO;

  double tmpval;

  /* */

  MAT = mxGetPr(prhs[0]);
  n   = mxGetN (prhs[0]);

  subMAT = mxGetPr(prhs[1]);
  k      = mxGetN (prhs[1]);

  plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
  projMAT = mxGetPr(plhs[0]);

  /* */

  MYCALLOC(smallMAT,      double, k*k);
  MYCALLOC(TempMatrix_nk, double, n*k);
  MYCALLOC(TempMatrix_kk, double, k*k);

  MYCALLOC(EigVals,  double,   k);
  MYCALLOC(EigVecs,  double, k*k);
  MYCALLOC(SuppEigVals, size_t, 2*k);

  /*
   *
   * STAGE 1
   *
   *   Calculate smallMAT = subMAT' * MAT * subMAT.
   *
   *          MAT is n x n
   *       subMAT is n x k
   *     smallMAT is k x k
   *  
   *     TempMatrix_nk = MAT * subMAT (via dsymm_)
   *
   *     smallMAT = subMAT' * TempMatrix_nk (via dgemm_)
   *
   *       Ideally, could exploit symmetry of smallMAT, but BLAS/LAPACK
   *       do not seem to have such a specialized procedure.
   *
   * STAGE 2
   *
   *   Calculate spectral decomposition smallMAT = EigVecs * Diag(
   *   EigVals ) * EigVecs'.
   *   
   * STAGE 3
   *
   *   Calculate final projMAT = subMAT * PosEigVecs * Diag( PosEigVals ) *
   *   PosEigVecs' * subMAT'.
   *
   *     TempMatrix_nk = subMAT * EigVecs (via dgemm_)
   *     
   *       Since size of EigVals is probably less than k, we do not use
   *       all columns of TempMatrix_nk.
   *
   *     projMAT = TempMatrix_nk * Diag( PosEigVals ) * TempMatrix_nk' (via dsyr_)
   *
   */


  /* STAGE 1 */
  /* STAGE 1 */
  /* STAGE 1 */

  dsymm_(&SIDE, &UPLO, &n, &k, &dOne, MAT, &n, subMAT, &n, &dZero, TempMatrix_nk, &n);

  dgemm_(&TRANST, &TRANSN, &k, &k, &n, &dOne, subMAT, &n, TempMatrix_nk, &n, &dZero, smallMAT, &k);

  /* STAGE 2 */
  /* STAGE 2 */
  /* STAGE 2 */

  /* Setup temporary space. */

  dWorkLength = -1;
  iWorkLength = -1;

  dsyevr_(&JOBZ, &RANGE, &UPLO, &k, TempMatrix_kk, &k, &VL, &VU, &IL,
          &IU, &ABSTOL, &NumEigs, EigVals, EigVecs, &k, SuppEigVals,
          &temp_dWork, &dWorkLength, &temp_iWork, &iWorkLength, &INFO);

  if (INFO != 0) 
    mexErrMsgTxt("project: Problem with dsyevr_ (first stage).");

  dWorkLength = (size_t) temp_dWork;
  iWorkLength = mymax(10*k, temp_iWork);

  if (dWorkLength > 0)
    MYCALLOC(dWork, double, dWorkLength);
  if (iWorkLength > 0)
    MYCALLOC(iWork, size_t, iWorkLength);

  /* Calculate range [VL,VU] for eigenvalues that we want. */

  /* VL = 0 and VU is a trivial upper bound on the largest positive
     eigenvalue. */

  VL = 0.0;
  VU = 0.0;
  for(i = 0; i < k*k; i++)
    VU += smallMAT[i]*smallMAT[i];
  VU = sqrt(VU);

  /* if( ABSTOL < VU && VU < 1.0 ) {
    tmpval = 1/VU;
    for(i = 0; i < k*k; i++)
      smallMAT[i] *= tmpval;
    VU = 1.0;
  }
  else tmpval = 1.0; */

  /* Copy smallMAT into TempMatrix_kk. */

  for(i = 0; i < k*k; i++)
    TempMatrix_kk[i] = smallMAT[i];

  /* Do spectral decomposition. */

  dsyevr_(&JOBZ, &RANGE, &UPLO, &k, TempMatrix_kk, &k, &VL, &VU, &IL,
          &IU, &ABSTOL, &NumEigs, EigVals, EigVecs, &k, SuppEigVals,
          dWork, &dWorkLength, iWork, &iWorkLength, &INFO);

  /* Early on, I had some problems with dsyevr_ failing. I found it
   * could be alleviated by computing all eigenvalues, not just the
   * nonnegative ones. So the following is a hack. I don't even know
   * if it gets called anymore. */

  if (INFO != 0) {
    
    /* printf("Hey! VU = %f\n", VU); */

    for(i = 0; i < k*k; i++)
      TempMatrix_kk[i] = smallMAT[i];

    dWorkLength = -1;

    dsyev_(&JOBZ, &UPLO, &k, TempMatrix_kk, &k, EigVals, &temp_dWork, &dWorkLength, &INFO);

    if (INFO != 0) 
      mexErrMsgTxt("project: Problem with dsyevr_ (second stage).");

    dWorkLength = (size_t) temp_dWork;

    if (dWorkLength > 0) {
      if (dWork != NULL) MYFREE(dWork);
      MYCALLOC(dWork, double, dWorkLength);
    }

    dsyev_(&JOBZ, &UPLO, &k, TempMatrix_kk, &k, EigVals, dWork, &dWorkLength, &INFO);

    if (INFO != 0)
      mexErrMsgTxt("project: Problem with dsyevr_ (second stage).");

    for(i = 0; i < k*k; i++)
      EigVecs[i] = TempMatrix_kk[i];

    /* for(i = 0; i < k*k; i++)
      printf("%e\n", smallMAT[i]);  */
    /* tmpval = 0.0;
    for(i = 0; i < k*k; i++)
      tmpval += pow(smallMAT[i],2); */

    /* printf("Hey! INFO = %d   tmpval = %e  VU = %e\n", INFO, tmpval, VU); */

    /* for(i = 0; i < k*k; i++)
      TempMatrix_kk[i] = smallMAT[i]; */

    /* VL = -VU;
    dsyevr_(&JOBZ, &RANGE, &UPLO, &k, TempMatrix_kk, &k, &VL, &VU, &IL,
            &IU, &ABSTOL, &NumEigs, EigVals, EigVecs, &k, SuppEigVals,
            dWork, &dWorkLength, iWork, &iWorkLength, &INFO); */

    /* for(j = 0; j < NumEigs; j++)
      printf("%e\n", EigVals[j]); */

    /* if (INFO != 0)
      mexErrMsgTxt("project: Problem with dsyevr_ (second stage)."); */

  }

  /* for(j = 0; j < NumEigs; j++)
    EigVals[j] /= tmpval; */


  /* STAGE 3 */
  /* STAGE 3 */
  /* STAGE 3 */

  dgemm_(&TRANSN, &TRANSN, &n, &NumEigs, &k, &dOne, subMAT, &n, EigVecs, &k, &dZero, TempMatrix_nk, &n);

  /* Initialize projMAT to 0. */

  for(i = 0; i < n*n; i++)
    projMAT[i] = 0.0;

  /* projMAT */

  for(j = 0; j < NumEigs; j++)
    if (EigVals[j] >= 0.0) {
      dsyr_(&UPLO, &n, &(EigVals[j]), TempMatrix_nk + j*n, &iOne, projMAT, &n);
    }

  /* Symmetrize projMAT */

  for(i = 0; i < n; i++)
    for(j = 0; j < i; j++)
      projMAT[i + j*n] = projMAT[j + i*n];


  MYFREE(smallMAT);
  MYFREE(TempMatrix_nk);
  MYFREE(TempMatrix_kk);

  MYFREE(EigVals);
  MYFREE(EigVecs);
  MYFREE(SuppEigVals);

  MYFREE(dWork);
  MYFREE(iWork);

  return;

}
