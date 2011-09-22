%mex -largeArrayDims -O opt_dnn_subprob_Y.c
%mex -largeArrayDims -O opt_dnn_subprob_Z.c -lmwlapack -lmwblas
mex -largeArrayDims opt_dnn_subprob_Y.c -lmwlapack -lmwblas
mex -largeArrayDims opt_dnn_subprob_Z.c -lmwlapack -lmwblas

%% Possible Windows variation
%% In Windows, do not append underscores to blas function names in .c files.

%% mex -O project.c C:\Progra~1\MATLAB\R2008a\extern\lib\win32\lcc\libmwlapack.lib C:\Progra~1\MATLAB\R2008a\extern\lib\win32\lcc\libmwblas.lib 


% mex -largeArrayDims -I/user/biz/jieqchen/matlab/cplex110/include/ilcplex -L/user/biz/jieqchen/matlab/cplex110/lib/x86-64_sles9.0_3.3/static_pic cplexint.c -lcplex
