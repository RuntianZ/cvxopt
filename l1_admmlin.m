function [x1, out1] = l1_admmlin(x0, A, b, mu, opts1)
%l1_admmlin - ADMM with linearization for primal
% opts1: [t k delta]
% t - Step size (default = 4)
% k - Number of iterations (default = 100)
% delta - Proximal term (default = 0.005)
% Step size rule: Fixed step size
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with ADMM with linearization...\n');
l = length(opts1);
if l >= 1
    t = opts1(1);
else
    t = 4;
end
if l >= 2
    k = opts1(2);
else
    k = 100;
end
if l >= 3
    delta = opts1(3);
else
    delta = 0.005;
end
assert(t>0);
assert(k>0);
ra = (A'*A+delta*eye(n))^(-1);
x1 = x0;
z = zeros(m,1);
y = (t*A*x1-t*b)/(1+t);
ostar = inf;

for i = 1:k
    g = A'*(A*x1-b)+mu*sign(sign(x1)+0.5);
    x1 = ra*(A'*(y+b+z/t)-g/t);
    z = z-t*(A*x1-y-b);
    y = (t*A*x1-t*b-z)/(1+t);
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

x1 = xstar;
out1 = ostar;
fprintf('ADMM with linearization complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end