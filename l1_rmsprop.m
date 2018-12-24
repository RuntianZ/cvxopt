function [x1, out1] = l1_rmsprop(x0, A, b, mu, opts1)
%l1_rmsprop - RMSProp with subgradient
% opts1: [t rho k delta]
% t - Step size (default = 0.0005)
% rho - Decay rate (default = 0.9)
% k - Number of iterations (default = 1000)
% delta - For numerical stability (default = 1e-6)
% Step size rule: Fixed step size
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with RMSProp...\n');
x1 = x0;
l = length(opts1);
if l >= 1
    t = opts1(1);
else
    t = 0.0005;
end
if l >= 2
    rho = opts1(2);
else
    rho = 0.9;
end
if l >= 3
    k = opts1(3);
else
    k = 1000;
end
if l >= 4
    delta = opts1(4);
else
    delta = 1e-6;
end
assert(k>1);
ostar = inf;
r = zeros(n,1);

for iter = 1:k
    g = A'*(A*x1-b)+mu*sign(sign(x1)+0.5);
    r = rho*r+(1-rho)*g.*g;
    dx = t*g./sqrt(delta+r);
    
    % Update
    x1 = x1-dx;
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

out1 = ostar;
x1 = xstar;
fprintf('RMSProp complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end