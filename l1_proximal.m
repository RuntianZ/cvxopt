function [x1, out1] = l1_proximal(x0, A, b, mu, opts1)
%l1_proximal - Proximal gradient method
% opts1: [k]
% k - Number of iterations (default = 100)
% Step size rule: Fixed step (t = 1/L)
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with proximal gradient method...\n');
x1 = x0;
l = length(opts1);
if l >= 1
    k = opts1(1);
else
    k = 100;
end
assert(k>1);
L = norm(A'*A,2);
t = 1/L;
ostar = inf;

for iter = 1:k
    z = x1-t*A'*(A*x1-b);
    z1 = z;
    z = z+t*mu;
    z1 = z1-t*mu;
    z(z>0) = 0;
    z1(z1<0) = 0;
    x1 = z+z1;
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

out1 = ostar;
x1 = xstar;
fprintf('Proximal gradient method complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end