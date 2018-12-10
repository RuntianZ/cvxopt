function [x1, out1] = l1_smooth(x0, A, b, mu, opts1)
%l1_smooth - Gradient method for smoothed primal
% opts1: [lambda k]
% lambda - Parameter of smoothing (default = 10)
% k      - Number of iterations (default = 1000)
% Step size rule: BB
% Smoother: Huber penalty
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with smoothed primal: gradient method...\n');
x1 = x0;
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
    g1 = x1;
    g1(x1>lambda) = 1;
    g1(x1<-lambda) = -1;
    g1((x1<=lambda)&(x1>=-lambda))=g1((x1<=lambda)&(x1>=-lambda))/lambda;
    g1 = g1+A'*(A*x1-b);
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
fprintf('Smoothed primal: gradient method complete.\n');
fprintf('Number of Iterations: %d.\n', i);
fprintf('Optimal value: %.4f\n\n', out1);
end