function [trial_accept, L_accept, trial_count, Gamma_k, oracle] = ...
    fun_run_upfugs_standard_outer_iteration_smoothed(A, B, d, lambda, mu, x_prev, xtil_prev, ...
    xbar_prev, L_prev, Gamma_prev, alg)
% fun_run_upfugs_standard_outer_iteration_smoothed
% Run one standard outer iteration of UPFGS for the smoothed inner term.

    oracle = fun_init_upfugs_oracle_counts();
    L_trial = L_prev;
    trial_count = 0;
    total_inner_iters = 0;

    while true
        trial_count = trial_count + 1;
        gamma_k = fun_compute_gamma(L_trial, Gamma_prev);

        x_under_outer = (1 - gamma_k) * xbar_prev + gamma_k * x_prev;
        [g_xlb, grad_g] = fun_eval_smooth_part_and_gradient(B, d, lambda, x_under_outer);
        oracle.g_eval = oracle.g_eval + 1;
        oracle.g_grad = oracle.g_grad + 1;

        state = struct();
        state.x_prev = x_prev;
        state.xtil_prev = xtil_prev;
        state.xbar_prev = xbar_prev;
        state.gamma = gamma_k;
        state.L_trial = L_trial;
        state.M0_trial = max(alg.M0, L_trial);

        trial = fun_run_upfugs_inner_subproblem_smoothed(A, mu, grad_g, state, alg);
        oracle.f_eval = oracle.f_eval + trial.f_eval_count;
        oracle.f_subgrad = oracle.f_subgrad + trial.f_subgrad_count;
        total_inner_iters = total_inner_iters + trial.T;

        g_xbar = fun_eval_smooth_part(B, d, lambda, trial.xbar);
        oracle.g_eval = oracle.g_eval + 1;

        g_rhs = g_xlb + grad_g' * (trial.xbar - x_under_outer) ...
            + 0.5 * L_trial * norm(trial.xbar - x_under_outer, 1)^2;

        if g_xbar <= g_rhs + alg.ineq_tol
            trial_accept = trial;
            trial_accept.gamma = gamma_k;
            trial_accept.accepted_inner_iters = trial.T;
            trial_accept.total_inner_iters = total_inner_iters;
            L_accept = L_trial;
            Gamma_k = L_accept * gamma_k^2;
            return;
        end

        L_trial = 2 * L_trial;
    end
end
