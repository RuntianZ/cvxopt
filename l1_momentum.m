function [x1, out1] = l1_momentum(x0, A, b, mu, opts1)
%l1_momentum - Subgradient method with momentum
% opts1: [t alpha k]
% t - Step size (default = 0.2)
% k - Number of iterations (default = 1000)
% delta - For numerical stability (default = 1e-7)
% Step size rule: Fixed step size
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with subgradient method with momentum...\n');
x1 = x0;
l = length(opts1);
if l >= 1
    t = opts1(1);
else
    t = 1e-4;
end
if l >= 2
    alpha = opts1(2);
else
    alpha = 0.95;
end
if l >= 3
    k = opts1(3);
else
    k = 1000;
end
assert(k>1);
ostar = inf;
r = zeros(n,1);
v = 0;

for iter = 1:k
    g = A'*(A*x1-b)+mu*sign(sign(x1)+0.5);
    v = alpha*v-t*g;
    % Update
    x1 = x1+v;
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

out1 = ostar;
x1 = xstar;
fprintf('Subgradient method with momentum complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end