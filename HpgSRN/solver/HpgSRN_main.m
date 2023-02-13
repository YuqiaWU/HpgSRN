%% *************************************************************
% Filename: HpgSRN_main.m
%% *************************************************************
% This file is to solve the optimal problem:
% \min f(Ax) +\lambda * \|x\|_q^q,
% where A \in R^{m*n}, f is a loss function of generalized linear model.
%% *************************************************************

%% *************************************************************
function [out] = HpgSRN_main(prob,options)

main_tstart = clock; % record the start time

if isfield(prob,'floss'); floss = prob.floss;                                       end
if isfield(prob,'lam');   lam   = prob.lam;                                         end
if isfield(prob,'q');     q     = prob.q;                                           end
if isfield(prob,'m');     m     = prob.m;                                           end
if isfield(prob,'n');     n     = prob.n;                                           end

if isfield(options,'x0'); x0    = options.x0;  else; x0 = zeros(size(prob.A,2), 1); end
if isfield(options,'maxiter');         maxiter        = options.maxiter;            end
if isfield(options,'iter_print');      iter_print     = options.iter_print;         end
if isfield(options,'result_print');    result_print   = options.result_print;       end
if isfield(options,'tol');             tol            = options.tol;                end
if isfield(options,'r');               r              = options.r;                  end
if isfield(options,'Ini_step');        Ini_step= options.Ini_step;else; Ini_step =1;end
if isfield(options,'Anorm');           Anorm          = options.Anorm;              end


 if (iter_print)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************* HpgSRN for the L_q regularized generalized linear model **************');
    fprintf('\n ****************************************************');
    fprintf('*******************************************');
    fprintf('\n  iter   optmeasure  obj_diff  time ls-1 ls-2 sec-step cosangle pd nnz');
 end

%% **************** Initialization ****************
tau = 10;
cg_num = 500;

sec_time = 0;  % Total time used for second-order step
sec_iter = 0;  % Number of iterations entering second-order part
x = x0; 
times_for_same_supp = 0;

iterate.nnz = n;
[fobjold, grad_fxnew] = floss(x, prob, iterate);
obj_old = fobjold + lam*sum(abs(x).^q);
%% ************************ Main loop *******************************
 for iter = 1:maxiter
    pd = -1; % If Step 2 is not run, pd = -1; If Hessian is positive definite, pd = 1. Otherwise, pd = 0.
    angle = 1;
    %% *************** first order step ***********************************
    grad_fx = grad_fxnew;
    search_num1 = 0; 
    if iter == 1
        mu = Ini_step;
    else
        tk = (sk'*rk)/(sk'*sk); 
        mu = min(max(tk,1.0e-20),1.0e20);
    end
    
    while 1
        gk = x -(1/mu)*grad_fx;
        xbar = subp_solver(gk, lam, mu, q);
        iter_bar.J = find(abs(xbar)>0); iter_bar.nnz = length(iter_bar.J); 
        [robjf2] = floss(xbar, prob, iter_bar);
        obj_new = robjf2 + lam*sum(abs(xbar).^q);
        if obj_new <= obj_old - 1.0e-8*norm(x-xbar)
            xnew = xbar;
            break
        else
            mu = tau * mu;
            search_num1 = search_num1+1;
        end
    end
    
    J_new = find(abs(xnew) > 0);
    nnz = length(J_new);
    iterate.J = J_new; 
    iterate.nnz = nnz;

    
    %% ******************** Judgement of entering Step 2 ***********************
    if iter > 1 && length(J) == length(J_new)
        if all(J == J_new) && all(sign(x(J)) == sign(xnew(J_new)))
            times_for_same_supp = times_for_same_supp + 1;    
        else
            times_for_same_supp = 0;
        end
    else
        times_for_same_supp = 0;
    end

    cond = iter>1 && times_for_same_supp >= r;
    if cond == 1
        u = x(J_new);
        minx = min(abs(u));
        omegax = mu + lam*q*(q-1)* minx^(q-2);
        omegaxbar = mu + lam*q*(q-1)*min(abs(xnew(J_new)))^(q-2);
        if omegax<0.1*omegaxbar
            cond = 0;
        end
    end

%     opt_measure = norm(x-xbar, 'inf')*mu;
%     if opt_measure < tol
%         cond = 0;
%     end

    %% ***************************************************************
    
%% ******************* second order step ********************
    search_num2 = 0;
    if cond
        sec_start = clock;
        N_step = 1; 
        
        reduce_prob = prob;
        reduce_prob.A = prob.A(:, J);
        reduce_prob.m = m;
        reduce_prob.n = nnz;          
        reduce_iter.nnz = nnz;
        cg_flag = ((nnz>cg_num)&&(times_for_same_supp<10))||nnz>3000;
        reduce_prob.cg_flag = cg_flag;
        
      
        
        [robjf, rgradf, rhessf] = floss(u, reduce_prob, reduce_iter);
        rgradg = lam * q * abs(u).^(q-1).* sign(u);
        rhessg = lam*q * (q-1) * abs(u).^(q-2);
        rgradF = rgradf+rgradg;
       
        if ~cg_flag
            rhessF = rhessf + diag(rhessg);
            try 
                L1 = chol(rhessF);
                L2 = -L1'\rgradF;
                ds = L1\L2;
                pd=1;
            catch
                upper_bound = sum(abs(diag(rhessF)));
                temp_devi = rhessF + upper_bound * eye(nnz);
                small_eig = eigs(temp_devi, 1, 'SM','Tolerance', 1.0e-8)- upper_bound;
                if isnan(small_eig) || small_eig>0
                    small_eig = min(eig(rhessF));
                end
                norm_gradF = norm(rgradF)^(0.5);
                reg_hessF = rhessF - (1.0+1.0e-8) * small_eig * eye(nnz) + 1.0e-3 * norm_gradF * eye(nnz);
                try 
                    L1 = chol(reg_hessF);
                    L2 = -L1'\rgradF;
                    ds = L1\L2;
                    pd = 0;
                catch
                    ds = -reg_hessF\rgradF;
                end
            end
        else
            [ds] = cg_alg(rgradF, rhessg, rhessf); 
        end
            
        
        % J_old = J_new;
        % B = A(:, J_new); 
        ut =  u + N_step * ds;
        rgradF_ds = 1.0e-4 * rgradF' * ds;
        robjF0 = robjf + lam*sum(abs(u).^q);
        robjF1 = robjF0 + rgradF_ds;
        [robjf2] = floss(ut, reduce_prob, reduce_iter);
        robjF2 = robjf2 + lam*sum(abs(ut).^q);
        while robjF2 > robjF1
            N_step = N_step * 1/2;
            ut = u+N_step*ds;
            robjf2 = floss(ut, reduce_prob, reduce_iter);
            robjF2 = robjf2 +lam*sum(abs(ut).^q);
            robjF1 = robjF0 + N_step * rgradF_ds;
            search_num2 = search_num2+1; % count the line search time
        end
        obj_new = robjF2;
        xnew = zeros(n, 1);
        xnew(J_new) = ut;

        sec_iter = sec_iter + 1;
        sec_time = sec_time + etime(clock, sec_start); % Cumulative time for second order step
        angle = (-rgradF_ds * 1.0e4)/(norm(rgradF) * norm(ds));
    end
 %% ***************************************************************************************

 %%  Judgement of termination condition and recording some other information 

    obj_diff = obj_new - obj_old;
    obj_old = obj_new;
    
    % opt_measure = norm(x-xbar, 'inf')*mu;
    gam = Anorm/0.95;
    xbar_opt = subp_solver(x-1/gam*grad_fx, lam, gam, q);
    opt_measure = norm(x-xbar_opt, 'inf')*gam;
    iter_time = etime(clock, main_tstart);
                   
    if (iter_print)&&(mod(iter,1)==0)
        nnz = length(J_new);
        fprintf('\n %3d     %3.2e   %3.2e  %.3f  %i    %i    %.3f     %i  %i',iter,opt_measure,obj_diff,iter_time, search_num1, search_num2, angle, pd, nnz);
    end
    
    if (opt_measure<tol)
        out.xopt = xnew;
        out.cput = etime(clock, main_tstart);
        out.nnz = nnz;
        out.iter = iter;
        out.obj = obj_new;
        out.gradF = norm(x- xbar);
        out.sec_iter = sec_iter;
        out.sec_time = sec_time;
        out.solve_ok = 1;
        if result_print
            fprintf('\n*************************** Result printed by HpgSRN *******************************');
            fprintf('\n | iter | sec_iter | total time | sec time |  nnz  | norm(gradF) |   obj');
            fprintf('\n    %i      %i       %g        %g      %i     %e   %e', iter, sec_iter, out.cput, sec_time, nnz,out.gradF, obj_new);
            fprintf('\n**********************************************************************************');
        end
        return;
    end
    u = xnew(J_new);
    [~, grad_fxnew] = floss(xnew, prob, iterate);
    sk = xnew - x; rk = grad_fxnew - grad_fx; % parameters for BB rule
    x = xnew; 
    J = J_new;
end
if (iter==maxiter)
    out.xopt = xnew;
    out.cput = etime(clock, main_tstart);
    out.nnz = nnz;
    out.iter = iter;
    out.obj = obj_new;
    % out.gradF = norm(x- xbar);
    out.sec_iter = sec_iter;
    out.sec_time = sec_time;
    out.solve_ok = 0;
    if result_print
        fprintf('\n The algorithm cannot achieve the required precision in the maximal iteration!');
    end
    return;
end

