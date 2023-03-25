% This file is to compute the objective value, gradient and Hessian matrix
% of the loss function: f(x) = \sum_i (Ax-b)_i^2



function fun = Leastsquare_Loss()

fun = @(x, prob,iterate) Call_Leastsquare_Loss(x,prob,iterate);

end


function [fval, grad, hess] = Call_Leastsquare_Loss(x,prob,iterate)

A = prob.A;
b = prob.b;
m = prob.m;
n = prob.n;

if isfield(iterate,'nnz') 
    if isfield(iterate,'J') 
        J = iterate.J; 
    end
    nnz = iterate.nnz; 
else
    nnz= n;
end


if  n > 5000 && nnz/n < 0.01
    B = A(:, J);
    u = x(J);
    loss = B * u - b;
    fval = 0.5*norm(loss,2)^2;
    if nargout >= 2
        grad = A' * loss;
        if nargout >= 3 
            if prob.cg_flag==0
                hess = A'*A;
            else
                hess = @(x) Call_cg_Leastsquare(A, x);
            end
        end
    end
else
    loss = A*x-b;
    fval = 0.5*norm(loss,2)^2;
    if nargout >= 2
        grad = A' * loss;
        if nargout >= 3 
            if prob.cg_flag==0
                hess = A'*A;
            else
                hess = @(x) Call_cg_Leastsquare(A, x);
            end
        end
    end
end
end

function [cg_fun] = Call_cg_Leastsquare(A, x)

    Ax = A*x;
    cg_fun = A'*Ax;
    
end
