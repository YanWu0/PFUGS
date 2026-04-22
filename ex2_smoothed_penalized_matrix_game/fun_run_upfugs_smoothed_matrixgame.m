function res = fun_run_upfugs_smoothed_matrixgame(A, B, d, lambda, mu, alg, phi_star, x0)
% fun_run_upfugs_smoothed_matrixgame
% Run PFUGS for the smoothed matrix-game objective
%   Phi(x) = lambda*||B*x-d||^2 + mu*log(sum_j exp(<A_j,x>/mu)).

    if nargin < 8
        n = size(A, 1);
        switch lower(alg.x0_type)
            case 'uniform'
                x0 = ones(n, 1) / n;
            otherwise
                error('Unknown x0_type: %s', alg.x0_type);
        end
    end
    if nargin < 7
        phi_star = [];
    end

    x_prev = x0;
    xtil_prev = x0;
    xbar_prev = x0;
    L_prev = NaN;
    Gamma_prev = NaN;
    phi0 = fun_eval_smoothed_objective(A, B, d, lambda, mu, x0);
    has_phi_star = ~isempty(phi_star);
    if has_phi_star
        gap0 = phi0 - phi_star;
    else
        gap0 = NaN;
    end

    phi_bar = zeros(alg.maxit, 1);
    gap_bar = zeros(alg.maxit, 1);
    L_hist = zeros(alg.maxit, 1);
    gamma_hist = zeros(alg.maxit, 1);
    inner_iter_hist = zeros(alg.maxit, 1);
    outer_trial_hist = zeros(alg.maxit, 1);
    time_hist_sec = zeros(alg.maxit, 1);
    oracle_f_eval_hist = zeros(alg.maxit, 1);
    oracle_f_subgrad_hist = zeros(alg.maxit, 1);
    oracle_g_eval_hist = zeros(alg.maxit, 1);
    oracle_g_grad_hist = zeros(alg.maxit, 1);

    oracle = struct('g_eval', 0, 'g_grad', 0, 'f_eval', 0, 'f_subgrad', 0);
    stop_reason = 'maxit';
    k_done = 0;
    t_start = tic;
    if isfield(alg, 'print_first')
        print_first = alg.print_first;
    else
        print_first = 1;
    end

    if alg.verbose
        if has_phi_star
            fprintf('PFUGS-s iter %5d | gap = %.8e\n', 0, gap0);
        else
            fprintf('PFUGS-s iter %5d | phi(bar_x_k) = %.8e\n', 0, phi0);
        end
    end

    for k = 1:alg.maxit
        if k == 1
            [trial_accept, L_accept, trial_count, Gamma_k, oracle_k] = ...
                fun_run_upfugs_first_outer_iteration_smoothed(A, B, d, lambda, mu, ...
                x_prev, xtil_prev, xbar_prev, alg);
        else
            [trial_accept, L_accept, trial_count, Gamma_k, oracle_k] = ...
                fun_run_upfugs_standard_outer_iteration_smoothed(A, B, d, lambda, mu, ...
                x_prev, xtil_prev, xbar_prev, L_prev, Gamma_prev, alg);
        end

        oracle.g_eval = oracle.g_eval + oracle_k.g_eval;
        oracle.g_grad = oracle.g_grad + oracle_k.g_grad;
        oracle.f_eval = oracle.f_eval + oracle_k.f_eval;
        oracle.f_subgrad = oracle.f_subgrad + oracle_k.f_subgrad;

        x_prev = trial_accept.x;
        xtil_prev = trial_accept.xtil;
        xbar_prev = trial_accept.xbar;
        L_prev = L_accept;
        Gamma_prev = Gamma_k;
        k_done = k;

        phi_bar_k = fun_eval_smoothed_objective(A, B, d, lambda, mu, xbar_prev);
        if has_phi_star
            gap_k = phi_bar_k - phi_star;
        else
            gap_k = NaN;
        end

        phi_bar(k) = phi_bar_k;
        gap_bar(k) = gap_k;
        L_hist(k) = L_accept;
        gamma_hist(k) = trial_accept.gamma;
        inner_iter_hist(k) = trial_accept.total_inner_iters;
        outer_trial_hist(k) = trial_count;
        time_hist_sec(k) = toc(t_start);
        oracle_f_eval_hist(k) = oracle.f_eval;
        oracle_f_subgrad_hist(k) = oracle.f_subgrad;
        oracle_g_eval_hist(k) = oracle.g_eval;
        oracle_g_grad_hist(k) = oracle.g_grad;

        if has_phi_star
            should_print = alg.verbose && (k <= print_first ...
                || mod(k, alg.print_every) == 0 ...
                || k == alg.maxit ...
                || gap_k <= alg.epsilon + alg.ineq_tol ...
                || (isfield(alg, 'max_runtime_sec') && time_hist_sec(k) >= alg.max_runtime_sec));
            if should_print
                fprintf(['PFUGS-s iter %5d | gap = %.8e | L_k = %.3e | ' ...
                    'outer_trials = %d | accepted_T = %d | total_inner_iters = %d\n'], ...
                    k, gap_k, L_accept, trial_count, ...
                    trial_accept.accepted_inner_iters, trial_accept.total_inner_iters);
            end
        else
            should_print = alg.verbose && (k <= print_first ...
                || mod(k, alg.print_every) == 0 ...
                || k == alg.maxit ...
                || (isfield(alg, 'max_runtime_sec') && time_hist_sec(k) >= alg.max_runtime_sec));
            if should_print
                fprintf(['PFUGS-s iter %5d | phi(bar_x_k) = %.8e | L_k = %.3e | ' ...
                    'outer_trials = %d | accepted_T = %d | total_inner_iters = %d\n'], ...
                    k, phi_bar_k, L_accept, trial_count, ...
                    trial_accept.accepted_inner_iters, trial_accept.total_inner_iters);
            end
        end

        if has_phi_star && gap_k <= alg.epsilon + alg.ineq_tol
            stop_reason = 'epsilon';
            break;
        end
        if isfield(alg, 'max_runtime_sec') && time_hist_sec(k) >= alg.max_runtime_sec
            stop_reason = 'time_limit';
            break;
        end
    end

    res = struct();
    res.x_final = x_prev;
    res.xtil_final = xtil_prev;
    res.xbar_final = xbar_prev;
    res.L_final = L_prev;
    res.Gamma_final = Gamma_prev;
    res.k_done = k_done;
    res.stop_reason = stop_reason;
    res.phi_star = phi_star;
    res.phi0 = phi0;
    res.gap0 = gap0;
    res.epsilon = alg.epsilon;
    res.ineq_tol = alg.ineq_tol;
    res.runtime_sec = toc(t_start);
    if k_done > 0
        res.avg_time_per_iter_sec = res.runtime_sec / k_done;
    else
        res.avg_time_per_iter_sec = NaN;
    end
    res.oracle = oracle;
    res.phi_bar = phi_bar(1:k_done);
    res.gap_bar = gap_bar(1:k_done);
    res.L_hist = L_hist(1:k_done);
    res.gamma_hist = gamma_hist(1:k_done);
    res.inner_iter_hist = inner_iter_hist(1:k_done);
    res.outer_trial_hist = outer_trial_hist(1:k_done);
    res.time_hist_sec = time_hist_sec(1:k_done);
    res.oracle_hist = struct();
    res.oracle_hist.f_eval = oracle_f_eval_hist(1:k_done);
    res.oracle_hist.f_subgrad = oracle_f_subgrad_hist(1:k_done);
    res.oracle_hist.g_eval = oracle_g_eval_hist(1:k_done);
    res.oracle_hist.g_grad = oracle_g_grad_hist(1:k_done);
end
