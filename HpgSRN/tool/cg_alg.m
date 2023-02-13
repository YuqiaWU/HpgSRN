%% -------------------
%% File name: cg_alg.m
%% --------------------
% This file is to find the Newton direction by using conjugate gradient
% method.

function [d] = cg_alg(rgradF, rhessg, cg_fun)

d = zeros(length(rhessg), 1);
para_list = [0, 0.001, 0.01, 0.1, 1];
res_array = [];
for i = 1:5
    min_eig = min(rhessg);
    reg_para = (- (para_list(i)+1.0e-8) * min_eig + 1.0e-4 * norm(rgradF)^(0.5) );
    res = rgradF;
    res_array(1) = norm(res);
    p = -res;
    cg_iter = 0;
    while 1
        cg_iter = cg_iter+ 1;
        Ap = cg_fun(p) + rhessg .* p + reg_para * p;
        pAp = p' * Ap;
        alpha = sum(abs(res).^2)/pAp;
        d_new = d + alpha * p;
        res_new = res + alpha * Ap;
        res_array = [res_array; norm(res_new)];
        % Termination condition
        if norm(res_new) < 1.0e-2 * norm(rgradF, 'inf')
            d = d_new;
            break;
        end
        if (cg_iter > 5) && rgradF' * d_new < 0
            d = d_new;
            break;
        end
        if (cg_iter > 8)
            break;
        end
        beta = sum(abs(res_new).^2)/sum(abs(res).^2);
        p_new = -res_new + beta*p;
        
        % update the parameter
        d = d_new; p = p_new; res = res_new;
    end
    gd = rgradF' * d;
    if  gd > 0 && i == 5
        d = -rgradF;
    elseif gd < 0
        break;
    end
end 
