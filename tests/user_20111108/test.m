load A2
load b2
load Aeq
load beq
load c
load F
load x_0
load x_L
load x_U
opsiyon=struct('max_time',360000000);
opsiyon.use_single_processor = 1;
opsiyon.verbosity = 3;
[x,fValCheck,ExitFlag,Output,Lambda] = quadprog  (F,c,[],[],Aeq(1:end-1,:),beq(1:end-1),x_L,x_U,x_0);
[x,fValCheck,time,status]            = quadprogbb(F,c,[],[],Aeq(1:end-1,:),beq(1:end-1),x_L,x_U,opsiyon);
