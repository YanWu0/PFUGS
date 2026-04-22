function [j_k, g_k] = fun_compute_full_subgradient(A, B, d, lambda, x)
% fun_compute_full_subgradient
% Compute a subgradient of
%   Phi(x) = lambda * ||B*x - d||_2^2 + max_j <A_j, x>
% at x.

    scores = A' * x;
    [~, j_k] = max(scores);
    g_k = 2 * lambda * (B' * (B * x - d)) + A(:, j_k);
end
