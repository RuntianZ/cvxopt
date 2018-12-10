function [x1, out1] = l1_fistab(x0, A, b, mu, opts1)
%l1_fistab - FISTA: Basic version
% opts1: [k]
% k - Number of iterations (default = 1000)
% Step size rule: Fixed step (t = 1/L)
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with FISTA: basic version...\n');
x1 = x0;
x2 = x1;
l = length(opts1);
if l >= 1
    k = opts1(1);
else
    k = 1000;
end
assert(k>1);
L = norm(A'*A,2);
t = 1/L;
ostar = inf;

for i = 1:k
    y = x1+(i-2)/(i+1)*(x1-x2);
    z = y-t*A'*(A*y-b);
    z1 = z;
    z = z+t*mu;
    z1 = z1-t*mu;
    z(z>0) = 0;
    z1(z1<0) = 0;
    x2 = x1;
    x1 = z+z1;
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

out1 = ostar;
x1 = xstar;
fprintf('FISTA: basic version complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end