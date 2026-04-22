function M_tilde_f_ub = fun_compute_entropy_smooth_constant_smoothed_max_upper(A, mu)
% fun_compute_entropy_smooth_constant_smoothed_max_upper
% Computable global upper bound for the smoothness constant of
%   \tilde f(x) = mu * log(sum_j exp(<A_j, x> / mu))
% with respect to the l1 primal norm and linfty dual norm.
%
% Using |Cov(U,V)| <= range(U) * range(V) / 4, we obtain
%   M_tilde_f <= (max_i Delta_i)^2 / (4 * mu),
% where Delta_i = max_j A_{ij} - min_j A_{ij}.

    row_max = full(max(A, [], 2));
    row_min = full(min(A, [], 2));
    row_range = row_max - row_min;
    M_tilde_f_ub = (max(row_range)^2) / (4 * mu);
end
