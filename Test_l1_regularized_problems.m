% function Test_l1_regularized_problems

% min 0.5 ||Ax-b||_2^2 + mu*||x||_1

% generate data
n = 1024;
m = 512;

A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;

mu = 1e-3;

x0 = zeros(n,1);

errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));

% cvx calling mosek
opts1 = []; %modify options
tic; 
[x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

% cvx calling gurobi
opts2 = []; %modify options
tic; 
[x2, out2] = l1_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;

% call mosek directly
opts3 = []; %modify options
tic; 
[x3, out3] = l1_mosek(x0, A, b, mu, opts3);
t3 = toc;

% call gurobi directly
opts4 = []; %modify options
tic; 
[x4, out4] = l1_gurobi(x0, A, b, mu, opts4);
t4 = toc;

% other approaches

% Projected gradient method
x0 = zeros(n,1);
opts5 = [1e-6]; %modify options
tic; 
[x5, out5] = l1_pgd(x0, A, b, mu, opts5);
t5 = toc;

% Subgradient method
x0 = zeros(n,1);
opts6 = [1000]; %modify options
tic; 
[x6, out6] = l1_subgradient(x0, A, b, mu, opts6);
t6 = toc;

% Gradient method for smoothed primal
x0 = zeros(n,1);
opts7 = [5e-4 1500]; %modify options
tic; 
[x7, out7] = l1_smooth(x0, A, b, mu, opts7);
t7 = toc;

% Gradient method for smoothed primal (decreasing lambda)
x0 = zeros(n,1);
opts8 = [0.1 1000]; %modify options
tic; 
[x8, out8] = l1_smoothd(x0, A, b, mu, opts8);
t8 = toc;

% Fast gradient method for smoothed primal
x0 = zeros(n,1);
opts9 = [0.1 1000]; %modify options
tic; 
[x9, out9] = l1_smoothf(x0, A, b, mu, opts9);
t9 = toc;

% Proximal gradient method
x0 = zeros(n,1);
opts10 = [1000]; %modify options
tic; 
[x10, out10] = l1_proximal(x0, A, b, mu, opts10);
t10 = toc;

% FISTA: Basic version
x0 = zeros(n,1);
opts11 = [1500 1.3]; %modify options
tic; 
[x11, out11] = l1_fistab(x0, A, b, mu, opts11);
t11 = toc;

% FISTA: Descent version
x0 = zeros(n,1);
opts12 = [1500 1.3]; %modify options
tic; 
[x12, out12] = l1_fistad(x0, A, b, mu, opts12);
t12 = toc;

% Nesterov's 2nd method
x0 = zeros(n,1);
opts13 = [1500 1.3]; %modify options
tic; 
[x13, out13] = l1_nes2(x0, A, b, mu, opts13);
t13 = toc;

% ADMM for dual
x0 = zeros(n,1);
opts15 = []; %modify options
tic; 
[x15, out15] = l1_admm(x0, A, b, mu, opts15);
t15 = toc;

% ADMM with linearization
x0 = zeros(n,1);
opts16 = [4 100 0.005]; %modify options
tic; 
[x16, out16] = l1_admmlin(x0, A, b, mu, opts16);
t16 = toc;

% Subgradient with momentum
x0 = zeros(n,1);
opts17 = [1e-3 0.995 3000]; %modify options
tic; 
[x17, out17] = l1_momentum(x0, A, b, mu, opts17);
t17 = toc;

% AdaGrad
x0 = zeros(n,1);
opts18 = [0.2 1000]; %modify options
tic; 
[x18, out18] = l1_adagrad(x0, A, b, mu, opts18);
t18 = toc;

% RMSProp
x0 = zeros(n,1);
opts19 = []; %modify options
tic; 
[x19, out19] = l1_rmsprop(x0, A, b, mu, opts19);
t19 = toc;

% Adam
x0 = zeros(n,1);
opts20 = [0.001 0.5 0.9]; %modify options
tic; 
[x20, out20] = l1_adam(x0, A, b, mu, opts20);
t20 = toc;

% print comparison results with cvx-call-mosek
fprintf('     cvx-call-gurobi: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t2, errfun(x1, x2));
fprintf('          call-mosek: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t3, errfun(x1, x3));
fprintf('         call-gurobi: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t4, errfun(x1, x4));
fprintf('                 PGD: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t5, errfun(x1, x5));
fprintf('         Subgradient: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t6, errfun(x1, x6));
fprintf('   Smoothed gradient: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t7, errfun(x1, x7));
fprintf(' Smoothed decreasing: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t8, errfun(x1, x8));
fprintf('       Fast smoothed: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t9, errfun(x1, x9));
fprintf('            Proximal: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t10, errfun(x1, x10));
fprintf('         FISTA-Basic: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t11, errfun(x1, x11));
fprintf('       FISTA-Descent: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t12, errfun(x1, x12));
fprintf('        Nesterov 2nd: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t13, errfun(x1, x13));
fprintf('       ADMM for dual: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t15, errfun(x1, x15));
fprintf('    ADMM with linear: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t16, errfun(x1, x16));
fprintf('Subgradient momentum: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t17, errfun(x1, x17));
fprintf('             AdaGrad: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t18, errfun(x1, x18));
fprintf('             RMSProp: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t19, errfun(x1, x19));
fprintf('                Adam: cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', t20, errfun(x1, x20));