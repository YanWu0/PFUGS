function sol = fun_compute_optval_qp(A, B, d, lambda, solver)
% fun_compute_optval_qp
% ------------------------------------------------------------
% Solve the convex QP reformulation
%
%   min_{x in Delta_n} lambda * ||B*x - d||_2^2 + max_j <A_j, x>
%
% by introducing an epigraph variable t:
%
%   min_{x,t} lambda * ||B*x - d||_2^2 + t
%   s.t.      A' * x <= t * 1,
%             x >= 0,
%             1' * x = 1.

    [n, m] = size(A);

    H = blkdiag(2 * lambda * (B' * B), 0);
    f = [-2 * lambda * (B' * d); 1];

    Aineq = [A', -ones(m,1)];
    bineq = zeros(m,1);

    Aeq = [ones(1,n), 0];
    beq = 1;

    lb = [zeros(n,1); -inf];
    ub = [];

    quadprog_path = which('quadprog');
    if contains(lower(quadprog_path), 'mosek')
        % MOSEK's quadprog wrapper does not accept optimoptions objects.
        options = [];
    else
        options = optimoptions('quadprog', ...
            'Display', solver.display, ...
            'OptimalityTolerance', solver.tol_fun, ...
            'StepTolerance', solver.tol_x, ...
            'MaxIterations', solver.max_iter);
    end

    solver_timer = tic;
    [z_star, fval, exitflag, output] = quadprog(H, f, Aineq, bineq, Aeq, beq, lb, ub, [], options); %#ok<ASGLU>
    runtime_sec = toc(solver_timer);

    x_star = z_star(1:n);
    t_star = z_star(end);

    quad_term = lambda * norm(B * x_star - d)^2;
    max_term = max(A' * x_star);
    phi_star = quad_term + max_term; % this is the optimal value that we need

    sol = struct();
    sol.x_star = x_star;
    sol.t_star = t_star;
    sol.phi_star = phi_star;
    sol.quad_term = quad_term;
    sol.max_term = max_term;
    sol.qp_obj = fval + lambda * (d' * d);
    sol.exitflag = exitflag;
    sol.output = output;
    sol.runtime_sec = runtime_sec;
end
