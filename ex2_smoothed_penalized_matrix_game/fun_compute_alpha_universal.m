function alpha = fun_compute_alpha_universal(M_curr, A_prev)
% fun_compute_alpha_universal
% Solve
%   M_curr * alpha^2 = A_prev * (1 - alpha)
% for the positive root alpha in (0,1].

    disc = A_prev^2 + 4 * M_curr * A_prev;
    alpha = (-A_prev + sqrt(disc)) / (2 * M_curr);
end
