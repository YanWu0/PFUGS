function alpha = fun_compute_alpha_ags(A_prev)
% fun_compute_alpha_ags
% Solve alpha^2 = A_prev * (1 - alpha) for the positive root in (0,1].

    disc = A_prev^2 + 4 * A_prev;
    alpha = (-A_prev + sqrt(disc)) / 2;
end
