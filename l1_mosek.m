function [x1, out1] = l1_mosek(x0, A, b, mu, opts1)
%l1_mosek - Calling mosek directly
n = size(A);
m = n(1);
n = n(2);

clear prob;
prob.c = [mu*ones(2*n,1); zeros(m,1)];

prob.qosubi = ((2*n+1):(2*n+m))';
prob.qosubj = ((2*n+1):(2*n+m))';
prob.qoval  = ones(m,1);

prob.a = [A -A -eye(m)];
prob.blc = b;
prob.buc = b;
prob.blx = [zeros(2*n,1); -inf*ones(m,1)];
prob.bux = [];

[r,res] = mosekopt('minimize',prob);

sol = res.sol.itr;
x1 = sol.xx(1:n)-sol.xx((n+1):(2*n));
out1 = sol.pobjval;
end