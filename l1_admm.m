function [x1, out1] = l1_admm(x0, A, b, mu, opts1)
%l1_admm - ADMM for dual
% opts1: [t]
% t - Step size (default = 5)
% tol - Tolerance (default = 1e-8)
% Step size rule: Fixed step size
n = size(A);
m = n(1);
n = n(2);

fprintf('Optimizing with ADMM for dual...\n');
l = length(opts1);
if l >= 1
    t = opts1(1);
else
    t = 5;
end
if l >= 2
    tol = opts1(2);
else
    tol = 1e-8;
end
assert(t>0);
ra = (eye(m)+t*(A*A'))^(-1);
nu = A*x0-b;
w = A'*nu;
w(w>mu) = mu;
w(w<-mu) = -mu;
z = zeros(n,1);
iter_num = 0;

while 1 == 1
    nu = ra*(A*z+t*A*w-b);
    w = A'*nu;
    if sum(w>mu+tol)+sum(w<-mu-tol) == 0
        break
    end
    iter_num = iter_num+1;
%     if mod(iter_num,100) == 0
%         disp(iter_num)
%         disp(max(w))
%     end
    w = A'*nu-z/t;
    w(w>mu) = mu;
    w(w<-mu) = -mu;
    z = z-t*(A'*nu-w);
end

y = A'*nu;
d = n-m;
i = 0;
C = [A;zeros(d,n)];
while m < n
    i = i+1;
    if y(i)>-mu+(1e-6) && y(i)<mu-(1e-6)
        m = m+1;
        C(m,i) = 1;
    end
end

x1 = C^(-1)*[(b+nu);zeros(d,1)];
out1 = 0.5*norm(A*x1-b,2)^2+mu*norm(x1,1);
fprintf('ADMM for dual complete.\n');
fprintf('Number of iterations:%d.\n', iter_num);
fprintf('Optimal value: %.4f\n\n', out1);
end