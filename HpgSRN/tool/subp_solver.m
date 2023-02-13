%% This file is to solve the proximal mapping of Lq norm
%% min tau/2 ||x-g||^2 + lambda ||x||_q^q, q\in [0,1].

function xopt = subp_solver(g, lambda, tau, q)

if q == 1 || q == 0 || q==1/2 || q == 2/3
    switch q
        case 0
            lambda = lambda/tau;
            xopt = g.*sign(max(abs(g) - sqrt(2 * lambda), 0));
        case 1
            lambda = lambda/tau;
            xopt = sign(max(abs(g) - lambda, 0)).*(g - lambda.* sign(g));
        case 0.5
            n = length(g);
            xopt = zeros(n, 1);
            lambda = 2*lambda/tau;
            thre = 54^(1/3)/4 * (lambda)^(2/3);
            J = (abs(g) >= thre);
            gi = g(J);
            t = lambda/8 * (abs(gi)/3).^(-3/2);
            phi = acos(t);
            xopt(J) = 2/3 * gi .* (1 + cos(2 * pi/3 - 2/3 * phi));
        case 2/3
            n = length(g);
            xopt = zeros(n, 1);
            lambda = 2 * lambda/tau ;
            thre = 2/3*(3 * lambda^3)^(1/4);
            J = (abs(g) >= thre);
            gi = g(J);
            phi = acosh(27/16 * gi.^2 * lambda^(-3/2));
            t = 2/sqrt(3) * lambda^(1/4) * sqrt(cosh(phi/3));
            xopt(J) = sign(gi).* ((t + sqrt(2 * abs(gi)./t - t.^2))/2).^3;
    end
    
else
    %% for the case the proximal mapping does not have analytical solution, we use numerical method to solve this subproblem.
    lambda = lambda/tau;
    n = length(g);
    xt = (lambda * q * (1-q))^(1/(2-q)).* sign(g); % h''(xt) = 0, so that for each i, xbari \in (xt, g_i) 
    
    xtmp1 = xt;
    xtmp2 = g;
    obj1 = 1/2 * (xt - g).^2 + lambda* abs(xt).^q;
    obj2 = lambda* abs(g).^q;
    comp = sign(obj1-obj2);
    count = 0;
    while max(abs(xtmp1 - xtmp2)) > 1.0e-6 
        count = count + 1;
        term_new = max(abs(xtmp1 - xtmp2));
        idx1 = find(comp == 1);
        xtmp1(idx1) = (xtmp1(idx1)+xtmp2(idx1))/2;
        idx2 = setdiff((1:n), idx1);
        % idx2 = find(comp == -1);
        xtmp2(idx2) = (xtmp1(idx2)+xtmp2(idx2))/2;
        obj1 = 1/2 * (xtmp1 - g).^2 + lambda* abs(xtmp1).^q;
        obj2 = 1/2 * (xtmp2 - g).^2 + lambda* abs(xtmp2).^q;
        comp = sign(obj1-obj2);     
    end
    
    xopt = zeros(n, 1);
    obj_zero = 1/2 * (0 - g).^2;
    comp_final = sign(obj_zero-obj1);
    nz_idx = find(comp_final == 1);
    xopt(nz_idx) = (xtmp1(nz_idx)+xtmp2(nz_idx))/2;    
end