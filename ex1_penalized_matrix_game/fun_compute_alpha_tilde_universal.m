function alpha_tilde = fun_compute_alpha_tilde_universal(alpha)
% fun_compute_alpha_tilde_universal
% Return the positive root to
%   alpha_tilde^2 = alpha^2 * (1 - alpha_tilde).

    disc = alpha^4 + 4 * alpha^2;
    alpha_tilde = (-alpha^2 + sqrt(disc)) / 2;
end
