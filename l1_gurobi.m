function [x1, out1] = l1_gurobi(x0, A, b, mu, opts1)
%l1_gurobi - Calling gurobi directly
n = size(A);
m = n(1);
n = n(2);

model.Q = sparse((2*n+1):(2*n+m),(2*n+1):(2*n+m),0.5*ones(m,1));
model.obj = [mu*ones(2*n,1); zeros(m,1)];

model.A = sparse([A -A -eye(m)]);
model.rhs = b;
model.sense = repmat('=',1,m);
model.lb = [zeros(2*n,1); -inf*ones(m,1)];

res = gurobi(model);
x1 = res.x(1:n)-res.x((n+1):(2*n));
out1 = res.objval;

end