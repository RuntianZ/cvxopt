function [x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1)
%l1_cvx_mosek - Calling mosek with cvx
n = size(A);
n = n(2);
cvx_clear
cvx_solver mosek
cvx_begin 
    variable x(n)
    minimize (0.5*(A*x-b)'*(A*x-b)+mu*norm(x,1))
cvx_end
x1 = x;
out1 = cvx_optval;
end