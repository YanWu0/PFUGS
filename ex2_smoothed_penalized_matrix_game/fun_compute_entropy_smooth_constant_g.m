function L_g = fun_compute_entropy_smooth_constant_g(B, lambda)
% fun_compute_entropy_smooth_constant_g
% Exact smoothness constant of g(x) = lambda * ||B*x-d||_2^2
% with respect to the l1 primal norm and linfty dual norm:
%   L_g = 2 * lambda * ||B' * B||_{1->infty}
%       = 2 * lambda * max_{i,j} |(B' * B)_{ij}|.

    gram_B = B' * B;
    L_g = 2 * lambda * full(max(max(abs(gram_B))));
end
