clear

n = 5;
H = 2*rand(n) - 1;
H = 0.5*(H+H');
f = 2*rand(n,1) - 1;
A = [];
b = [];
Aeq = [];
beq = [];
LB = zeros(n,1);
UB = ones(n,1);

options.verbosity = 0;

[x,fval,time,stat] = quadprogbb(H,f,A,b,Aeq,beq,LB,UB,options);

load extra/ex2_1_1
options.use_single_processor = 0;
[x,fval,time,stat] = quadprogbb(H,f,A,b,Aeq,beq,LB,UB,options);
options.use_single_processor = 1;
[x,fval,time,stat] = quadprogbb(H,f,A,b,Aeq,beq,LB,UB,options);
