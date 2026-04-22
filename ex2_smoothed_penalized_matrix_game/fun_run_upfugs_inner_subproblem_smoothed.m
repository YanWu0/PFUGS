function trial = fun_run_upfugs_inner_subproblem_smoothed(A, mu, grad_g_outer, state, alg)
% fun_run_upfugs_inner_subproblem_smoothed
% Run the UPFGS inner line-search subroutine for the smoothed inner term.

    x_prev = state.x_prev;
    xtil_outer_prev = state.xtil_prev;
    xbar_outer_prev = state.xbar_prev;
    gamma = state.gamma;
    L_trial = state.L_trial;
    M0_trial = state.M0_trial;

    x_prev_t = x_prev;
    xtil_prev_t = xtil_outer_prev;
    M_prev = M0_trial;
    A_prev = NaN;
    c_prev = 1;
    used_special_c_prev = false;

    f_eval_count = 0;
    f_subgrad_count = 0;
    inner_backtracks = 0;

    for t = 1:alg.max_inner_it
        M_base = c_prev * M_prev;
        M_t = M_base;
        accepted = false;

        while ~accepted
            if t == 1
                alpha_t = 1;
            else
                alpha_t = fun_compute_alpha_universal(M_t, A_prev);
            end

            eta_t = L_trial * gamma;
            p_t = L_trial * gamma * (1 - alpha_t) / alpha_t + gamma * M_t * alpha_t;

            xlb_t = (1 - gamma) * xbar_outer_prev ...
                + gamma * ((1 - alpha_t) * xtil_prev_t + alpha_t * x_prev_t);
            [f_xlb, grad_f_t] = fun_eval_smoothed_max_part(A, xlb_t, mu);
            f_eval_count = f_eval_count + 1;
            f_subgrad_count = f_subgrad_count + 1;

            x_t = fun_entropy_two_reference_prox_update( ...
                x_prev, x_prev_t, grad_g_outer + grad_f_t, eta_t, p_t);
            xtil_t = (1 - alpha_t) * xtil_prev_t + alpha_t * x_t;
            xbar_t = (1 - gamma) * xbar_outer_prev + gamma * xtil_t;

            f_xbar = fun_eval_smoothed_max_part(A, xbar_t, mu);
            f_eval_count = f_eval_count + 1;

            rhs = f_xlb + grad_f_t' * (xbar_t - xlb_t) ...
                + 0.5 * M_t * norm(xbar_t - xlb_t, 1)^2 ...
                + 0.5 * gamma * alpha_t * alg.epsilon;

            if f_xbar <= rhs + alg.ineq_tol
                accepted = true;
            else
                M_t = 2 * M_t;
                inner_backtracks = inner_backtracks + 1;
            end
        end

        A_t = M_t * alpha_t^2;
        terminate_now = false;
        if t == 1
            terminate_now = abs(M_t - L_trial) <= alg.ineq_tol;
        elseif used_special_c_prev && abs(M_t - M_base) <= alg.ineq_tol
            terminate_now = true;
        end

        if terminate_now
            trial = struct();
            trial.x = x_t;
            trial.xtil = xtil_t;
            trial.xbar = xbar_t;
            trial.T = t;
            trial.M_last = M_t;
            trial.A_last = A_t;
            trial.alpha_last = alpha_t;
            trial.p_last = p_t;
            trial.f_eval_count = f_eval_count;
            trial.f_subgrad_count = f_subgrad_count;
            trial.inner_backtracks = inner_backtracks;
            return;
        end

        alpha_tilde = fun_compute_alpha_tilde_universal(alpha_t);
        if M_t * alpha_tilde^2 <= L_trial + alg.ineq_tol
            denom = (M_t * alpha_t^2 - L_trial)^2;
            c_t = L_trial * alpha_t^4 * M_t / max(denom, realmin);
            used_special_c_t = true;
        else
            c_t = 1;
            used_special_c_t = false;
        end

        x_prev_t = x_t;
        xtil_prev_t = xtil_t;
        M_prev = M_t;
        A_prev = A_t;
        c_prev = c_t;
        used_special_c_prev = used_special_c_t;
    end

    error('UPFGS smoothed inner subproblem reached max_inner_it without termination.');
end
