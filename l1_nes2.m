function [x1, out1] = l1_nes2(x0, A, b, mu, opts1)
%l1_nes2 - Nesterov's 2nd method
% opts1: [k r]
% k - Number of iterations (default = 1000)
% r - t = r/L (default = 1)
% Step size rule: Fixed step (theta = 2/(i+1), t = r/L)
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with Nesterov''s 2nd method...\n');
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
    z = v-t/theta*A'*(A*y-b);
    z1 = z;
    z = z+t*mu/theta;
    z1 = z1-t*mu/theta;
    z(z>0) = 0;
    z1(z1<0) = 0;
    v = z+z1;
    x1 = (1-theta)*x1+theta*v;
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

x1 = xstar;
out1 = ostar;
fprintf('FISTA: Nesterov''s 2nd method complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end