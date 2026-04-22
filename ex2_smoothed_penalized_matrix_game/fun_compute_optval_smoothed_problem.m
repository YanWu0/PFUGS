function sol = fun_compute_optval_smoothed_problem(A, B, d, lambda, mu, solver, x0)
% fun_compute_optval_smoothed_problem
% Solve
%   min_{x in Delta_n} lambda * ||B*x-d||_2^2 + mu*log(sum_j exp(<A_j,x>/mu))
% with a smooth constrained solver.

    n = size(A, 1);

    if nargin < 7 || isempty(x0)
        x0 = ones(n, 1) / n;
    end

    Aeq = ones(1, n);
    beq = 1;
    lb = zeros(n, 1);
    ub = ones(n, 1);

    obj = @(x) fun_eval_smoothed_objective_and_gradient(A, B, d, lambda, mu, x);

    options = optimoptions('fmincon', ...
        'Algorithm', 'interior-point', ...
        'Display', solver.display, ...
        'SpecifyObjectiveGradient', true, ...
        'OptimalityTolerance', solver.tol_fun, ...
        'StepTolerance', solver.tol_x, ...
        'MaxIterations', solver.max_iter);

    solver_timer = tic;
    [x_star, fval, exitflag, output] = fmincon(obj, x0, [], [], Aeq, beq, lb, ub, [], options); %#ok<ASGLU>
    runtime_sec = toc(solver_timer);

    [smooth_term, ~] = fun_eval_smooth_part_and_gradient(B, d, lambda, x_star);
    smooth_max_term = fun_eval_smoothed_max_part(A, x_star, mu);

    sol = struct();
    sol.x_star = x_star;
    sol.phi_star = fval;
    sol.quad_term = smooth_term;
    sol.smoothed_max_term = smooth_max_term;
    sol.exitflag = exitflag;
    sol.output = output;
    sol.runtime_sec = runtime_sec;
end
