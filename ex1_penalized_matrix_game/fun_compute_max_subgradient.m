function [f_val, j_star, subgrad] = fun_compute_max_subgradient(A, x)
% fun_compute_max_subgradient
% Evaluate
%   f(x) = max_j <A_j, x>
% and return an active subgradient A_{j_star}.

    scores = A' * x;
    [f_val, j_star] = max(scores);
    subgrad = A(:, j_star);
end
