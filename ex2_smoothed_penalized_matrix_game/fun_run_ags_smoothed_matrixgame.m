function res = fun_run_ags_smoothed_matrixgame(A, B, d, lambda, mu, alg, phi_star, x0)
% fun_run_ags_smoothed_matrixgame
% Run entropy-based AGS on the smoothed matrix-game problem.

    if nargin < 7
        phi_star = [];
    end
    if nargin < 8
        n = size(A, 1);
        x0 = ones(n, 1) / n;
    end

    n = size(A, 1);
    D_x_sq = log(n);
    if alg.L_guess <= alg.M_guess
        outer_component = 'g';
        inner_component = 'f';
        L_outer = alg.L_guess;
        M_inner = alg.M_guess;
    else
        outer_component = 'f';
        inner_component = 'g';
        L_outer = alg.M_guess;
        M_inner = alg.L_guess;
    end

    N_ref = max(1, ceil(2 * sqrt(L_outer * D_x_sq / alg.epsilon)));
    T = max(1, ceil(4 * sqrt(M_inner * D_x_sq / alg.epsilon) / N_ref));
    if isfield(alg, 'maxit') && ~isempty(alg.maxit)
        max_outer_it = alg.maxit;
    else
        max_outer_it = 30000000;
    end

    x_prev = x0;
    xtil_prev = x0;
    xbar_prev = x0;

    phi0 = fun_eval_smoothed_objective(A, B, d, lambda, mu, x0);
    has_phi_star = ~isempty(phi_star);
    if has_phi_star
        gap0 = phi0 - phi_star;
    else
        gap0 = NaN;
    end

    phi_bar = zeros(max_outer_it, 1);
    gap_bar = zeros(max_outer_it, 1);
    time_hist_sec = zeros(max_outer_it, 1);
    oracle_f_eval_hist = zeros(max_outer_it, 1);
    oracle_f_subgrad_hist = zeros(max_outer_it, 1);
    oracle_g_eval_hist = zeros(max_outer_it, 1);
    oracle_g_grad_hist = zeros(max_outer_it, 1);

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
            fprintf('AGS iter %5d | gap = %.8e\n', 0, gap0);
        else
            fprintf('AGS iter %5d | phi(bar_x_k) = %.8e\n', 0, phi0);
        end
    end

    for k = 1:max_outer_it
        if toc(t_start) >= alg.max_runtime_sec
            stop_reason = 'time_limit';
            break;
        end

        gamma_k = 2 / (k + 1);
        eta_k = 2 * L_outer / k;

        x_under = (1 - gamma_k) * xbar_prev + gamma_k * x_prev;
        if strcmp(outer_component, 'g')
            [~, grad_outer] = fun_eval_smooth_part_and_gradient(B, d, lambda, x_under);
            oracle.g_grad = oracle.g_grad + 1;
        else
            [~, grad_outer] = fun_eval_smoothed_max_part(A, x_under, mu);
            oracle.f_subgrad = oracle.f_subgrad + 1;
        end

        x_prev_t = x_prev;
        xtil_prev_t = xtil_prev;
        A_prev = NaN;
        time_limit_hit = false;

        for t = 1:T
            if toc(t_start) >= alg.max_runtime_sec
                time_limit_hit = true;
                break;
            end

            if t == 1
                alpha_t = 1;
                A_t = 1;
            else
                alpha_t = fun_compute_alpha_ags(A_prev);
                A_t = alpha_t^2;
            end

            p_t = (2 * L_outer * (1 - alpha_t) + 2 * M_inner * alpha_t^2) / (alpha_t * k);

            xltil_t = (1 - alpha_t) * xtil_prev_t + alpha_t * x_prev_t;
            xlb_t = (1 - gamma_k) * xbar_prev + gamma_k * xltil_t;

            if strcmp(inner_component, 'f')
                [~, grad_inner] = fun_eval_smoothed_max_part(A, xlb_t, mu);
                oracle.f_subgrad = oracle.f_subgrad + 1;
            else
                [~, grad_inner] = fun_eval_smooth_part_and_gradient(B, d, lambda, xlb_t);
                oracle.g_grad = oracle.g_grad + 1;
            end

            x_t = fun_entropy_two_reference_prox_update(x_prev, x_prev_t, grad_outer + grad_inner, eta_k, p_t);
            xtil_t = (1 - alpha_t) * xtil_prev_t + alpha_t * x_t;
            xbar_t = (1 - gamma_k) * xbar_prev + gamma_k * xtil_t;

            x_prev_t = x_t;
            xtil_prev_t = xtil_t;
            A_prev = A_t;

            if toc(t_start) >= alg.max_runtime_sec
                time_limit_hit = true;
                break;
            end
        end

        if time_limit_hit
            stop_reason = 'time_limit';
            break;
        end

        x_prev = x_prev_t;
        xtil_prev = xtil_prev_t;
        xbar_prev = xbar_t;
        k_done = k;

        phi_bar_k = fun_eval_smoothed_objective(A, B, d, lambda, mu, xbar_prev);
        if has_phi_star
            gap_k = phi_bar_k - phi_star;
        else
            gap_k = NaN;
        end

        phi_bar(k) = phi_bar_k;
        gap_bar(k) = gap_k;
        time_hist_sec(k) = toc(t_start);
        oracle_f_eval_hist(k) = oracle.f_eval;
        oracle_f_subgrad_hist(k) = oracle.f_subgrad;
        oracle_g_eval_hist(k) = oracle.g_eval;
        oracle_g_grad_hist(k) = oracle.g_grad;

        if has_phi_star
            should_print = alg.verbose && (k <= print_first ...
                || mod(k, alg.print_every) == 0 ...
                || k == max_outer_it ...
                || gap_k <= alg.epsilon + alg.ineq_tol ...
                || time_hist_sec(k) >= alg.max_runtime_sec);
            if should_print
                fprintf(['AGS iter %5d | gap = %.8e | L_guess = %.3e | ' ...
                    'M_guess = %.3e | outer = %s | inner = %s | T_k = %d\n'], ...
                    k, gap_k, alg.L_guess, alg.M_guess, outer_component, inner_component, T);
            end
        else
            should_print = alg.verbose && (k <= print_first ...
                || mod(k, alg.print_every) == 0 ...
                || k == max_outer_it ...
                || time_hist_sec(k) >= alg.max_runtime_sec);
            if should_print
                fprintf(['AGS iter %5d | phi(bar_x_k) = %.8e | L_guess = %.3e | ' ...
                    'M_guess = %.3e | outer = %s | inner = %s | T_k = %d\n'], ...
                    k, phi_bar_k, alg.L_guess, alg.M_guess, outer_component, inner_component, T);
            end
        end

        if has_phi_star && gap_k <= alg.epsilon + alg.ineq_tol
            stop_reason = 'epsilon';
            break;
        end
        if time_hist_sec(k) >= alg.max_runtime_sec
            stop_reason = 'time_limit';
            break;
        end
    end

    res = struct();
    res.x_final = x_prev;
    res.xtil_final = xtil_prev;
    res.xbar_final = xbar_prev;
    res.k_done = k_done;
    res.stop_reason = stop_reason;
    res.N_ref = N_ref;
    res.T_used = T;
    res.outer_component = outer_component;
    res.inner_component = inner_component;
    res.L_outer_used = L_outer;
    res.M_inner_used = M_inner;
    res.L_guess = alg.L_guess;
    res.M_guess = alg.M_guess;
    res.phi_star = phi_star;
    res.phi0 = phi0;
    res.gap0 = gap0;
    res.epsilon = alg.epsilon;
    res.ineq_tol = alg.ineq_tol;
    res.runtime_sec = toc(t_start);
    res.oracle = oracle;
    res.phi_bar = phi_bar(1:k_done);
    res.gap_bar = gap_bar(1:k_done);
    res.time_hist_sec = time_hist_sec(1:k_done);
    res.oracle_hist = struct();
    res.oracle_hist.f_eval = oracle_f_eval_hist(1:k_done);
    res.oracle_hist.f_subgrad = oracle_f_subgrad_hist(1:k_done);
    res.oracle_hist.g_eval = oracle_g_eval_hist(1:k_done);
    res.oracle_hist.g_grad = oracle_g_grad_hist(1:k_done);
end
