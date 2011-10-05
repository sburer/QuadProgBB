%
% [LB,Y,Z,S,sig,ret] = opt_dnn(Q,c,A,b,B,E,L,U,max_iter,S,sig,LB_target,LB_beat,max_time,cons)
%
% ------------------------- ! IMPORTANT NOTES ! ------------------------
%
%   At this time, finite bounds U are required!
%
%   All inputs are required but may be entered as [] for default values!
%
%   All vectors are assumed to be column vectors!
%
%   Very little error checking at this time!
%
% ----------------------------------------------------------------------
%
% Solves the doubly nonnegative relaxation of
% 
%   min  0.5*x'*Q*x + c'*x
%   s.t. A*x == b
%          x >= 0
%          x(j) binary for all j in B
%          x(i)*x(j) = 0 for all (i,j) in E
%
% by the augmented Lagrangian method described in the accompanying
% paper.
%
% (Q,c,A,b) is entered in dense matrix/vector form. Sparsity is ignored
% at this time.
%
% Let n be the dimension of x. B is a 0-1 vector of size n with 1's
% indicating the binary positions in x. E is a symmetric 0-1 matrix of
% size n x n with 1's indicting the complementary pairs.
%
% L and U are vectors of additional simple bounds on x. In particular,
% L is assumed to be nonnegative.
%
% The remaining parameters are:
%
%   max_iter = the maximum number of augmented Lagrangian iterations
%   allowed; default 1000
%
%   S = initial dual multiplier; symmetric (1+n)x(1+n); default 0
%
%   sig = initial penalty parameter; positive scalar; default is max(
%   max(max(abs(0.5*Q))), max(abs(c)) )
%
%   LB_target = algorithm will terminate if the lower bound exceeds this
%   value; useful for fathoming when called during branch-and-bound
%
%   LB_beat = algorithm does not terminate until the lower bound exceeds this
%   value; useful for making sure child node solves as well as parent withink
%   branch-and-bound
% 
%   max_time = time limit (in seconds)
%
% The ouputs are:
%
%   LB  = best lower bound obtained
%   Y,Z = internal primal variables
%   S   = final dual multiplier
%   sig = final penalty parameter
%   ret = internal return code
%

function [LB,Y,Z,S,sig,ret] = opt_dnn(Q,c,A,b,B,E,L,U,max_iter,S,sig,LB_target,LB_beat,max_time,cons,verb)

% save mysave

%% ------------------------------------------------
%% Make sure subproblem routines have been compiled
%% ------------------------------------------------

if exist('opt_dnn_subprob_Y') ~= 3 | exist('opt_dnn_subprob_Z') ~= 3
  opt_dnn_compile
end

%% ------------------------------
%% Set internal algorithm options
%% ------------------------------

num_loop = 1 ; % Default number of block-coordinate descent
               % loops to do per aug Lag iter
bnd_freq = 25; % Uidate the lower bound every bnd_freq iters

    norm_tol = 1.0e-15; % Terminate if norm(Y-Z,'fro') < norm_tol
  change_tol = 1.0e-15; % Terminate if avg rel change in bound over 5 
                       % is less than change_tol
def_max_iter = 10000;  % Default max number of aug Lag iters;
                       % overridden by user's max_iter if present

%% --------------------------------------
%% Make sure Matlab is using only one CPU
%% --------------------------------------

% maxNumCompThreads(1);

%% -----------------------------
%% Set default values for inputs
%% -----------------------------

%% Note: Some very important quantities defined below such as n, S, C,
%% sig

if length(Q) == 0 & length(c) == 0
  error('Objective is empty!');
end

n = max(size(Q,1),length(c));

if length(Q) == 0
  Q = zeros(n);
end

if length(c) == 0
  c = zeros(n,1);
end

if length(A) == 0 | length(b) == 0
  warning('At least one of A or b is empty. Assuming both empty.');
  A = zeros(1,n);
  b = 0;
end

if length(B) == 0
  B = zeros(n,1);
end

if length(E) == 0
  E = zeros(n);
end

if length(L) == 0
  L = zeros(n,1);
end

if length(U) == 0
  U = Inf*ones(n,1);
end

if length(max_iter) == 0
  max_iter = def_max_iter;
end

if length(LB_target) == 0
  LB_target = Inf;
end

if length(S) == 0
  S = zeros(1+n);
end

max_iter_orig = max_iter; 

C = 0.5*[0,c';c,Q];
C = 0.5*full(C + C');

if length(sig) == 0
  sig = max(max(abs(C)));
  if sig == 0
    sig = 1;
  end
end
sig_orig = sig;

%% ------------------
%% Verify U is finite
%% ------------------

if sum(U) == Inf
  error('At this time, U must be finite.');
end

%% ---------------------------------
%% Calculate additional derived data
%% ---------------------------------

N = null([b,-A]);

L = [1;L];
U = [1;U];

B = [0;B];
E = [0,zeros(1,n);zeros(n,1),E];

%% -------------------------
%% Initialize various things
%% -------------------------

LB_curr = -Inf; % Current lower bound
     LB = -Inf; % Best lower bound obtained so far

rel_changes = zeros(5,1); % Vector storing changes in LB_curr relative to
                          % LB

       S_save = S;        % Saved copy of current S for later possible backtracking
num_loop_save = num_loop; % Ditto for num_loop
bnd_freq_save = bnd_freq; % Ditto for bnd_freq

Z = zeros(1+n); % Initial Z

start_cputime = cputime; % For timings

%% ----------------------------------
%% Run augmented Lagrangian algorithm
%% ----------------------------------

numwarns=0;

%% Do iterations that update S, sig

iter = 1;
while iter <= max_iter

  %% Do iterations that optimize aug Lag subproblem via block coordinate
  %% descent over Y and Z

  for loop = 1 : num_loop

    %% Minimize wrt Y

    [tmp,Y] = opt_dnn_subprob_Y(C - S - sig*Z, L, U, sig, B, E);

    %% Minimize wrt Z 

    Z = opt_dnn_subprob_Z(Y - (1/sig)*S, N);

  %% End iterations that optimize aug Lag subproblem

  end

  %% Update S

  S = S - sig*(Y - Z);
%   S = S + opt_dnn_subprob_Z(-S, N); %% Do not need this (from Hongbo)!

  %% Update LB_curr and sigma (every bnd_freq iterations)

  if mod(iter,bnd_freq) == 0

    %% Compute LB_curr

    [LB_curr,tmp] = opt_dnn_subprob_Y(C - S, L, U, 0, B, E);

    %% Calculate bound change ratio (used in two places below)

    ratio = 1 + (LB_curr - LB)/(1 + abs(LB));

    %% Update sigma (only if iter > bnd_freq because then we have
    %% a sensible ratio)
    %%
    %% Also allow for backtracking if sigma would be nonsensical

    if iter > bnd_freq

      if ratio <= 0 | sig * ratio <= sig_orig/100
        numwarns = numwarns+1;
        warning('Got possible sig <= 0 or sig <= sig_orig/100. Backtracking to fix.');
        S = S_save;
        num_loop = num_loop+1;
        bnd_freq = max(1,ceil(bnd_freq/2));
        if numwarns >= 5
          ret = 'poor';
          return
        end
      else
        S_save = S;
        sig = sig * ratio;
        num_loop = max(num_loop_save,num_loop-1);
        bnd_freq = min(bnd_freq_save,2*bnd_freq);
      end

    end

    %% Print information

    if verb == 2

      if mod(iter,bnd_freq*4) == 0  
        fprintf('iter = %4d   LB = %.8e   time = %4.2f   norm = %.1e   sig = %.1e\n', ...
         iter, max(LB,LB_curr)+cons, cputime - start_cputime, norm(Y-Z,'fro')/(0.5*(norm(Y,'fro')+norm(Z,'fro'))), sig);
      end
    
    elseif verb >= 3

         fprintf('iter = %4d   LB = %.8e   time = %4.2f   norm = %.1e   sig = %.1e\n', ...
         iter, max(LB,LB_curr)+cons, cputime - start_cputime, norm(Y-Z,'fro')/(0.5*(norm(Y,'fro')+norm(Z,'fro'))), sig);

    end

    %% If LB_curr is better than LB...

    if LB_curr > LB 

      %% Update relative change

      rel_changes(1:4) = rel_changes(2:5);
      rel_changes(5) = ratio;

      %% Update LB 

      LB = LB_curr;

      %% Possibly terminate by relative change in LB_curr (change_tol)
      %% (Make sure 5 updates have occurred)

      if min(rel_changes) >= 1 & mean(rel_changes) < 1 + change_tol
        ret = 'rel_change';
        return
      end

      %% Possibly terminate by fathoming (LB_target)

      if LB > LB_target
        ret = 'fathom';
        return
      end

    end

    %% Possibly terminate by norm of Y-Z (norm_tol)

    if norm(Y-Z,'fro')/(0.5*(norm(Y,'fro')+norm(Z,'fro'))) < norm_tol
      ret = 'norm';
      return
    end

    %% If LB_beat has not been beaten, then increase
    %% max_iter (by increment of max_iter_orig)

    if LB <= LB_beat & iter > max_iter - bnd_freq & max_iter < 10*max_iter_orig
      max_iter = max_iter + max_iter_orig;
    end

  end

%% End iterations that update S, sig

  iter = iter+1;

  if cputime - start_cputime > max_time
    ret = 'time';
    return
  end

end

if LB <= LB_beat
  ret = 'poor';
else
  ret = 'iter';
end

return
