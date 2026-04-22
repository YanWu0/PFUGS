function res = fun_run_fgm_matrixgame_benchmark(A, B, d, lambda, alg, phi_star, x0)
% fun_run_fgm_matrixgame_benchmark
% ------------------------------------------------------------
% Run the benchmark algorithm for
%   Phi(x) = lambda * ||B*x - d||_2^2 + max_j <A_j, x>
%
% Iteration:
%   underline{x}_k = (1-gamma_k) bar{x}_{k-1} + gamma_k x_{k-1}
%   g_k = 2*lambda*B'*(B*underline{x}_k - d) + A_{j_k}
%   x_k = argmin_{x in Delta_n} <g_k,x> + eta_k * V(x_{k-1},x)
%   bar{x}_k = (1-gamma_k) bar{x}_{k-1} + gamma_k x_k
%
% Acceptance test:
%   Phi(bar{x}_k) <= Phi(underline{x}_k) + <g_k, bar{x}_k-underline{x}_k>
%                    + (L/2)*||bar{x}_k - underline{x}_k||_1^2
%                    + (epsilon/2)*gamma_k.

    if nargin < 7
        n = size(A, 1);

        switch lower(alg.x0_type)
            case 'uniform'
                x0 = ones(n, 1) / n;
            otherwise
                error('Unknown x0_type: %s', alg.x0_type);
        end
    end
    if nargin < 6
        phi_star = [];
    end

    x_prev = x0; % x_{k-1}
    xbar_prev = x0; % \bar{x}_{k-1}
    L_prev = alg.L0; % L_{k-1}
    Gamma_prev = NaN; %\Gamma_{k-1}
    phi0 = fun_eval_objective(A, B, d, lambda, x0);
    has_phi_star = ~isempty(phi_star);
    if has_phi_star
        gap0 = phi0 - phi_star;
    else
        gap0 = NaN;
    end

    if isfield(alg, 'ineq_tol')
        ineq_tol = alg.ineq_tol;
    else
        ineq_tol = 1e-12;
    end

    phi_bar = zeros(alg.maxit, 1);
    phi_under = zeros(alg.maxit, 1);
    gap_bar = zeros(alg.maxit, 1);
    L_hist = zeros(alg.maxit, 1);
    gamma_hist = zeros(alg.maxit, 1);
    eta_hist = zeros(alg.maxit, 1);
    backtrack_count = zeros(alg.maxit, 1);
    max_index_hist = zeros(alg.maxit, 1);
    time_hist_sec = zeros(alg.maxit, 1);
    oracle_f_eval_hist = zeros(alg.maxit, 1);
    oracle_f_subgrad_hist = zeros(alg.maxit, 1);
    oracle_g_eval_hist = zeros(alg.maxit, 1);
    oracle_g_grad_hist = zeros(alg.maxit, 1);

    stop_reason = 'maxit';
    k_done = 0;
    oracle = struct();
    oracle.g_eval = 0;
    oracle.g_grad = 0;
    oracle.f_eval = 0;
    oracle.f_subgrad = 0;

    t_start = tic;
    if isfield(alg, 'print_first')
        print_first = alg.print_first;
    else
        print_first = 1;
    end

    if alg.verbose
        if has_phi_star
            fprintf('iter %5d | gap = %.8e\n', 0, gap0);
        else
            fprintf('iter %5d | phi(bar_x_k) = %.8e\n', 0, phi0);
        end
    end

    for k = 1:alg.maxit
        accepted = false;
        s = 0;
        if k == 1
            L_trial = L_prev;
        else
            L_trial = 0.5 * L_prev;
        end

        while ~accepted
            if k == 1
                gamma_k = 1;
                Gamma_k = L_trial;
            else
                gamma_k = fun_compute_gamma(L_trial, Gamma_prev);
                Gamma_k = L_trial * gamma_k^2;
            end

            eta_k = L_trial * gamma_k;
            x_under = (1 - gamma_k) * xbar_prev + gamma_k * x_prev;
            [j_k, g_k] = fun_compute_full_subgradient(A, B, d, lambda, x_under);
            x_k = fun_entropy_prox_update(x_prev, g_k, eta_k);
            xbar_k = (1 - gamma_k) * xbar_prev + gamma_k * x_k;

            phi_under_k = fun_eval_objective(A, B, d, lambda, x_under);
            phi_bar_k = fun_eval_objective(A, B, d, lambda, xbar_k);
            oracle.g_grad = oracle.g_grad + 1;
            oracle.f_subgrad = oracle.f_subgrad + 1;
            % FGM evaluates the two objective parts together through Phi,
            % so the value-evaluation counts of f and g should match.
            oracle.g_eval = oracle.g_eval + 2;
            oracle.f_eval = oracle.f_eval + 2;

            lhs = phi_bar_k;
            rhs = phi_under_k + g_k' * (xbar_k - x_under) ...
                + 0.5 * L_trial * norm(xbar_k - x_under, 1)^2 ...
                + 0.5 * alg.epsilon * gamma_k;

            if lhs <= rhs + ineq_tol
                accepted = true;
            else
                s = s + 1;
                L_trial = 2 * L_trial;
            end
        end

        if has_phi_star
            gap_k = phi_bar_k - phi_star;
        else
            gap_k = NaN;
        end

        x_prev = x_k;
        xbar_prev = xbar_k;
        L_prev = L_trial;
        Gamma_prev = Gamma_k;
        k_done = k;

        phi_bar(k) = phi_bar_k; % current obj-value
        phi_under(k) = phi_under_k;
        gap_bar(k) = gap_k; % current obj-value gap
        L_hist(k) = L_trial;
        gamma_hist(k) = gamma_k;
        eta_hist(k) = eta_k;
        backtrack_count(k) = s + 1;
        max_index_hist(k) = j_k; % max function active index
        time_hist_sec(k) = toc(t_start);
        oracle_f_eval_hist(k) = oracle.f_eval;
        oracle_f_subgrad_hist(k) = oracle.f_subgrad;
        oracle_g_eval_hist(k) = oracle.g_eval;
        oracle_g_grad_hist(k) = oracle.g_grad;

        if has_phi_star
            should_print = alg.verbose && (k <= print_first ...
                || mod(k, alg.print_every) == 0 ...
                || k == alg.maxit ...
                || gap_k <= alg.epsilon + ineq_tol);
            if should_print
                fprintf(['iter %5d | gap = %.8e | L_k = %.3e | trials = %d\n'], ...
                         k, gap_k, L_trial, s + 1);
            end

            if gap_k <= alg.epsilon + ineq_tol
                stop_reason = 'epsilon';
                break;
            end
        else
            should_print = alg.verbose && (k <= print_first ...
                || mod(k, alg.print_every) == 0 ...
                || k == alg.maxit);
            if should_print
                fprintf(['iter %5d | phi(bar_x_k) = %.8e | L_k = %.3e | trials = %d\n'], ...
                         k, phi_bar_k, L_trial, s + 1);
            end
        end
    end

    res = struct();
    res.x_final = x_prev;
    res.xbar_final = xbar_prev;
    res.L_final = L_prev;
    res.Gamma_final = Gamma_prev;
    res.k_done = k_done; % stopping iter k
    res.stop_reason = stop_reason;
    res.phi_star = phi_star;
    res.oracle = oracle;
    res.phi0 = phi0;
    res.gap0 = gap0;
    res.epsilon = alg.epsilon;
    res.ineq_tol = ineq_tol;
    res.runtime_sec = toc(t_start);
    if k_done > 0
        res.avg_time_per_iter_sec = res.runtime_sec / k_done;
    else
        res.avg_time_per_iter_sec = NaN;
    end

    res.phi_bar = phi_bar(1:k_done);
    res.phi_under = phi_under(1:k_done);
    res.gap_bar = gap_bar(1:k_done);
    res.L_hist = L_hist(1:k_done);
    res.gamma_hist = gamma_hist(1:k_done);
    res.eta_hist = eta_hist(1:k_done);
    res.backtrack_count = backtrack_count(1:k_done);
    res.max_index_hist = max_index_hist(1:k_done);
    res.time_hist_sec = time_hist_sec(1:k_done);
    res.oracle_hist = struct();
    res.oracle_hist.f_eval = oracle_f_eval_hist(1:k_done);
    res.oracle_hist.f_subgrad = oracle_f_subgrad_hist(1:k_done);
    res.oracle_hist.g_eval = oracle_g_eval_hist(1:k_done);
    res.oracle_hist.g_grad = oracle_g_grad_hist(1:k_done);
end
