% This file is to compute the objective value, gradient and Hessian matrix
% of the loss function: f(x) = \sum_i log(1+exp(-(Ax)_i)))



function fun = Logistics_Loss()

fun = @(x, prob,iterate) Call_Logistics_Loss(x,prob,iterate);

end


function [fval, grad, hess] = Call_Logistics_Loss(x,prob,iterate)

A = prob.A;
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


if  n > 5000 && nnz/n < 0.05
    B = A(:, J);
    u = x(J);
    loss = B * u;
    J1 = find(loss > 10); % to make power(e, temp(i)) well defined.
    J2 = setdiff((1:m), J1);
    fval = sum(log(1+exp(loss(J2)))) + sum(loss(J1));
    if nargout >= 2
        exp_loss = exp(loss);
        t = exp_loss./(1+exp_loss);
        s = t./(1+exp_loss);
        grad = A' * t;
        if nargout >= 3
            if prob.cg_flag==0
                one_vec = ones(1, nnz);
                tt = s*one_vec;
                temp = tt.*A;
                hess = A'*temp;
            else
                hess = @(x) Call_cg_Logistics(A, s, x);
            end
        end
    end
else
    loss = A * x;
    J1 = find(loss > 10);
    J2 = setdiff((1:m), J1);
    fval = sum(log(1+exp(loss(J2)))) + sum(loss(J1));
    if nargout >= 2
        exp_loss = exp(loss);
        t = exp_loss./(1+exp_loss);
        s = t./(1+exp_loss);
        grad = A' * t;
        if nargout >= 3
            if prob.cg_flag==0
                one_vec = ones(1, n);
                tt = s*one_vec;
                temp = tt.*A;
                hess = A'*temp;
            else
                hess = @(x) Call_cg_Logistics(A, s, x);
            end
        end
    end
end
end

function [cg_fun] = Call_cg_Logistics(A, s, x)

Ax = A*x;
sAx = s.*Ax;
cg_fun = A'*sAx;

end
