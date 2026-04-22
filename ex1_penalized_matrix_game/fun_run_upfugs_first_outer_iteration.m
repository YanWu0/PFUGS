function [trial_accept, L_accept, trial_count, Gamma_k, oracle] = ...
    fun_run_upfugs_first_outer_iteration(A, B, d, lambda, x_prev, xtil_prev, xbar_prev, alg)
% fun_run_upfugs_first_outer_iteration
% Run the special first outer iteration of PFUGS.

    oracle = fun_init_upfugs_oracle_counts(); % start counting from all 0s
    gamma_first = 1;
    trial_count = 0;
    total_inner_iters = 0;

    L_trial = alg.M0;
    while true
        trial_count = trial_count + 1;

        [g_xlb, grad_g] = fun_eval_smooth_part_and_gradient(B, d, lambda, x_prev);
        oracle.g_eval = oracle.g_eval + 1;
        oracle.g_grad = oracle.g_grad + 1;

        [f_xlb, j_k, subgrad_f] = fun_compute_max_subgradient(A, x_prev);
        oracle.f_eval = oracle.f_eval + 1;
        oracle.f_subgrad = oracle.f_subgrad + 1;

        x_trial = fun_entropy_two_reference_prox_update( ...
            x_prev, x_prev, grad_g + subgrad_f, L_trial, L_trial);
        g_xbar = fun_eval_smooth_part(B, d, lambda, x_trial);
        f_xbar = full(max(A' * x_trial));
        oracle.g_eval = oracle.g_eval + 1;
        oracle.f_eval = oracle.f_eval + 1;

        diff_vec = x_trial - x_prev;
        g_rhs = g_xlb + grad_g' * diff_vec + 0.5 * L_trial * norm(diff_vec, 1)^2;
        f_rhs = f_xlb + subgrad_f' * diff_vec ...
            + 0.5 * L_trial * norm(diff_vec, 1)^2 + 0.5 * alg.epsilon;

        if g_xbar <= g_rhs + alg.ineq_tol && f_xbar <= f_rhs + alg.ineq_tol
            first_accept = struct();
            first_accept.x = x_trial;
            first_accept.xtil = x_trial;
            first_accept.xbar = x_trial;
            first_accept.gamma = gamma_first;
            first_accept.accepted_inner_iters = 1;
            first_accept.active_index_last = j_k;
            L_first = L_trial;
            total_inner_iters = total_inner_iters + 1;
            break;
        end

        L_trial = 2 * L_trial;
    end

    trial_accept = first_accept;
    L_accept = L_first;

    L_trial = L_first / 2;
    while true
        trial_count = trial_count + 1;
        [g_xlb, grad_g] = fun_eval_smooth_part_and_gradient(B, d, lambda, x_prev);
        oracle.g_eval = oracle.g_eval + 1;
        oracle.g_grad = oracle.g_grad + 1;

        state = struct();
        state.x_prev = x_prev;
        state.xtil_prev = xtil_prev;
        state.xbar_prev = xbar_prev;
        state.gamma = 1;
        state.L_trial = L_trial;
        state.M0_trial = max(alg.M0, L_trial);

        trial = fun_run_upfugs_inner_subproblem(A, grad_g, state, alg);
        oracle.f_eval = oracle.f_eval + trial.f_eval_count;
        oracle.f_subgrad = oracle.f_subgrad + trial.f_subgrad_count;
        total_inner_iters = total_inner_iters + trial.T;

        g_xbar = fun_eval_smooth_part(B, d, lambda, trial.xbar);
        oracle.g_eval = oracle.g_eval + 1;

        g_rhs = g_xlb + grad_g' * (trial.xbar - x_prev) ...
            + 0.5 * L_trial * norm(trial.xbar - x_prev, 1)^2;

        if g_xbar <= g_rhs + alg.ineq_tol
            trial_accept = trial;
            trial_accept.gamma = 1;
            trial_accept.accepted_inner_iters = trial.T;
            L_accept = L_trial;
            L_trial = L_trial / 2;
        else
            break;
        end
    end

    trial_accept.total_inner_iters = total_inner_iters;
    Gamma_k = L_accept * trial_accept.gamma^2;
end
