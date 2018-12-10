function [x1, out1] = l1_pgd(x0, A, b, mu, opts1)
%l1_pgd - Projected Gradient Descent
% opts1: [tol k]
% tol  - Tolerance (default = 1e-4)
% k    - Max number of iterations (default = 300)
% Step size rule: BB
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with PGD...\n');
xp = max([x0 zeros(n)],[],2);   %x+
xm = max([-x0 zeros(n)],[],2);  %x-
x1 = [xp;xm];
l = length(opts1);
if l >= 1
    tol = opts1(1);
else
    tol = 1e-4;
end
if l >= 2
    k = opts1(2);
else
    k = 300;
end

iter_num = 0;
g1 = 0;
ostar = inf;
xt = 0;
while 1==1
    d1 = A'*(A*(xp-xm)-b);
    d2 = mu*ones(n,1);
    df_dxp = d1+d2;
    df_dxm = d2-d1;
    g2 = g1;
    g1 = [df_dxp;df_dxm];
    if iter_num==0
        lr = 0.0005;
    else
        lr = dot(x1-x2,g1-g2)/dot(g1-g2,g1-g2);
    end
    if isnan(lr) || lr == 0
        break
    end 
    iter_num = iter_num+1;
    
    xp = xp-lr*df_dxp;
    xm = xm-lr*df_dxm;
    xp(xp<0) = 0;
    xm(xm<0) = 0;
    x2 = x1;
    x1 = [xp;xm];
    if iter_num>=k
        break
    end
    xl = xt;
    xt = xp-xm;
    if iter_num > 1 && norm(xl-xt) < tol
        break
    end
    out1 = 0.5*norm(A*xt-b,2)^2+mu*norm(xt,1);
    if out1 < ostar
        ostar = out1;
        xstar = xt;
    end
end

x1 = xstar;
out1 = ostar;
fprintf('PGD complete.\nNumber of iterations: %d\n', iter_num);
fprintf('Optimal value: %.4f\n\n', out1);

end