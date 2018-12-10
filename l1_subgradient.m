function [x1, out1] = l1_subgradient(x0, A, b, mu, opts1)
%l1_subgradient - Subgradient method with k iterations
% opts1: [k]
% k - Number of iterations (default = 1000)
% Step size rule: Fixed length
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with subgradient method...\n');
x1 = x0;
l = length(opts1);
if l >= 1
    k = opts1(1);
else
    k = 1000;
end
assert(k>1);
R = norm(x0)+11.30;
s = R/sqrt(k);
g1 = 0;
ostar = inf;

for iter = 1:k
    % Computing subgradient
    g2 = g1;
    g1 = A'*(A*x1-b)+mu*sign(sign(x1)+0.5);
    
    % Find step size
    if iter == 1
        t = 0.0005;
    else
        t = s/norm(g2);
    end
    
    % Update
    x1 = x1-t*g1;
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

out1 = ostar;
x1 = xstar;
fprintf('Subgradient method complete.\n');
fprintf('Optimal value: %.4f\n\n', out1);
end