function [x1, out1] = l1_fistad(x0, A, b, mu, opts1)
%l1_fistad - FISTA: Descent version
% opts1: [k r]
% k - Number of iterations (default = 1000)
% r - t = r/L (default = 1)
% Step size rule: Fixed step (theta = 2/(i+1), t = r/L)
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with FISTA: descent version...\n');
x1 = x0;
v = x0;
l = length(opts1);
if l >= 1
    k = opts1(1);
else
    k = 1000;
end
if l >= 2
    r = opts1(2);
else
    r = 1;
end
assert(k>1);
L = norm(A'*A,2);
t = r/L;
ostar = inf;

for i = 1:k
    theta = 2/(i+1);
    y = (1-theta)*x1+theta*v;
    z = y-t*A'*(A*y-b);
    z1 = z;
    z = z+t*mu;
    z1 = z1-t*mu;
    z(z>0) = 0;
    z1(z1<0) = 0;
    u = z+z1;
    out1 = 0.5*norm(A*u-b,2)^2+mu*norm(u,1);
    v = x1+(u-x1)/theta;
    if out1 < ostar
        x1 = u;
        ostar = out1;
    end
end

out1 = ostar;
fprintf('FISTA: descent version complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end