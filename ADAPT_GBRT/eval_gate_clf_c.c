#include <math.h>
#include <string.h>
#include <time.h>
#include <mex.h>
#include "matrix.h"
#include <stdio.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  double *X, *Trees_gate, *Trees_clf, *preds_gate, *preds_clf, *cost, *tstCostp, *featureUsageFull;
  mxArray *featMat;
  bool *featMatp;
  double lr;
  int depth, interval;
  
  int d, ntst, ntrees;
  int treelen=0, index;

  int ii, jj, dd;
  /* first get the input data */
  /* input are X (nxd), Trees (2^0+2^1+...2^(depth-1))xntrees, depth, learningrate, interval */
  /* return predictions, nx1 */
  	
  X = mxGetPr(prhs[0]);
  ntst = mxGetM(prhs[0]);
  d = mxGetN(prhs[0]);
  Trees_gate = mxGetPr(prhs[1]);
  Trees_clf = mxGetPr(prhs[2]);
  depth = (int)(*(mxGetPr(prhs[3])));
  lr = (double)(*(mxGetPr(prhs[4])));
  cost= mxGetPr(prhs[5]); /*feature cost vector*/
  interval = (int)(*(mxGetPr(prhs[6])));
  
  /* individual tree length*/
  for (ii=0; ii<depth; ii++) {
  	  treelen += pow(2,ii);	
  }
  ntrees = (int)(mxGetM(prhs[1])/treelen);
  
  int totalpreds = 1;
  if (interval!=0) {
	  totalpreds = ntrees/interval;
  }
  /*mexPrintf("ntrees = %d, treelen = %d, totalpreds = %d\n", ntrees, treelen, totalpreds);*/
  /* Create output matrix */
  plhs[0] = mxCreateDoubleMatrix(ntst,totalpreds,mxREAL); 
  plhs[1] = mxCreateDoubleMatrix(ntst,totalpreds,mxREAL); 
  int dims[3];
  dims[0]= ntst; dims[1] = d; dims[2] = totalpreds;  
  plhs[2] = mxCreateLogicalArray(3,dims); /*to hold the history of feature used for each test example over trees  preds_gate = mxGetPr(plhs[0]);*/
  preds_gate = mxGetPr(plhs[0]);
  preds_clf = mxGetPr(plhs[1]);
  featMatp = mxGetLogicals(plhs[2]);
  memset(preds_gate,0,sizeof(double)*ntst*totalpreds);
  memset(preds_clf,0,sizeof(double)*ntst*totalpreds);
  
  /* evaluate trees*/
  int intind = 0;
  for (ii=0; ii<ntrees; ii++) {
	  if (interval!=0 && ii%interval==0 && ii>0) {
		  intind++;
	  }
	  int zxiitreelen = ii*treelen;
	  int oxiitreelen = ntrees*treelen+ii*treelen;
	  int txiitreelen = ntrees*treelen*3+ii*treelen;
	  for (jj=0; jj<ntst; jj++) {
		  /* evaluate each gating tree*/
		  int offset = 1;
		  for (dd=0; dd<depth-1; dd++) {
			  /* feature index*/
			  index = Trees_gate[zxiitreelen+offset-1]-1;
              featMatp[intind*ntst*d+ntst*index+jj]=true; /*update the feature indicator*/
			  if (X[ntst*index+jj] < Trees_gate[oxiitreelen+offset-1]) {
				  offset = offset*2;
			  }
			  else {
				  offset = offset*2+1;
			  }
		  }
		  if (interval!=0) {
              preds_gate[intind*ntst+jj] += lr*Trees_gate[txiitreelen+offset-1];
          }
          else{
              preds_gate[jj] += lr*Trees_gate[txiitreelen+offset-1];
          }
          
		  offset = 1;
		  for (dd=0; dd<depth-1; dd++) {
			  index = Trees_clf[zxiitreelen+offset-1]-1;
              featMatp[intind*ntst*d+ntst*index+jj]=true; /*update the feature indicator*/
			  if (X[ntst*index+jj] < Trees_clf[oxiitreelen+offset-1]) {
				  offset = offset*2;
			  }
			  else {
				  offset = offset*2+1;
			  }
		  }
		  if (interval!=0) {
              preds_clf[intind*ntst+jj] += lr*Trees_clf[txiitreelen+offset-1];
          }
          else{
              preds_clf[jj] += lr*Trees_clf[txiitreelen+offset-1];		          
          }
	  }
  }
  return;
}


	
