function [x1, out1] = l1_adam(x0, A, b, mu, opts1)
%l1_adam - Adam with subgradient
% opts1: [t rho1 rho2 k delta]
% t - Step size (default = 0.001)
% rho1 - First moment parameter (default = 0.9)
% rho2 - Second moment parameter (default = 0.999)
% k - Number of iterations (default = 1000)
% delta - For numerical stability (default = 1e-7)
% Step size rule: Fixed step size
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with Adam...\n');
x1 = x0;
l = length(opts1);
if l >= 1
    t = opts1(1);
else
    t = 0.001;
end
if l >= 2
    rho1 = opts1(2);
else
    rho1 = 0.9;
end
if l >= 3
    rho2 = opts1(3);
else
    rho2 = 0.999;
end
if l >= 4
    k = opts1(4);
else
    k = 1000;
end
if l >= 5
    delta = opts1(5);
else
    delta = 1e-7;
end
assert(k>1);
ostar = inf;
s = zeros(n,1);
r = zeros(n,1);
p1 = 1;
p2 = 1;

for iter = 1:k
    g = A'*(A*x1-b)+mu*sign(sign(x1)+0.5);
    s = rho1*s+(1-rho1)*g;
    r = rho2*r+(1-rho2)*g.*g;
    p1 = p1*rho1;
    p2 = p2*rho2;
    s = s/(1-p1);
    r = r/(1-p2);
    dx = t*s./(delta+sqrt(r));
    
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
fprintf('Adam complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end