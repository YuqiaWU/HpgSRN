%% Load data
load('colon');
load('colon_label');

%% Computation of data feature
[m, n] = size(colon);
b = colon_label;
A = colon' * diag(b);
A = A';

lambdamax = max(abs(sum(abs(A), 1)));
scale = 0.01;
lambda = scale * lambdamax;
q = 1/2;
eigsopt.issym = 1;
Rmap=@(x) A*x;
Rtmap=@(x) A'*x;
RRtmap=@(x) Rmap(Rtmap(x));
Anorm = eigs(RRtmap,length(b),1,'LA',eigsopt);

%% Algorithm

prob.A = A;
prob.b = 0;
prob.lam = lambda;
prob.q = q;
prob.m = m;
prob.n = n;
prob.floss = Logistics_Loss();

options.x0 = zeros(n,1);
options.tol = 1.0e-3;
options.iter_print = 1;
options.result_print = 1;
options.Ini_step =  1;
options.maxiter = 50000;
options.r = 4;
options.Anorm = 0.25*Anorm;

[out] = HpgSRN_main(prob,options); 
