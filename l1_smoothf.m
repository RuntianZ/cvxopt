function [x1, out1] = l1_smoothf(x0, A, b, mu, opts1)
%l1_smoothf - Fast gradient method for smoothed primal
% opts1: [lambda0 k]
% lambda0 - Initial value of lambda (default = 10)
% k       - Number of iterations (default = 1000)
% Step size rule: BB
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with smoothed primal: fast gradient method...\n');
x1 = x0;
x2 = x1;
l = length(opts1);
if l >= 1
    lambda = opts1(1);
else
    lambda = 10;
end
if l >= 2
    k = opts1(2);
else
    k = 1000;
end
assert(lambda>0);
assert(k>1);
ostar = inf;
g1 = 0;
for i = 1:k
    g2 = g1;
    y = x1+(i-2)/(i+1)*(x1-x2);
    g1 = y;
    j = lambda/i;
    g1(y>lambda/j) = 1;
    g1(y<-lambda/j) = -1;
    g1((y<=lambda/j)&(y>=-lambda)/j)=...
        j*g1((y<=lambda/j)&(y>=-lambda)/j)/lambda;
    g1 = g1+A'*(A*y-b);
    if i == 1
        t = 0.0005;
    else
        t = dot(x1-x2,g1-g2)/dot(g1-g2,g1-g2);
    end
    x2 = x1;
    x1 = x1-t*g1;
    out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
    if out1 < ostar
        xstar = x1;
        ostar = out1;
    end
end

out1 = ostar;
x1 = xstar;
fprintf('Smoothed primal: fast gradient method complete.\n');
fprintf('Number of Iterations: %d.\n', i);
fprintf('Optimal value: %.4f\n\n', out1);
end