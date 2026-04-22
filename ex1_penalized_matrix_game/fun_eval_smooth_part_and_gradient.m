function [g_val, grad_g] = fun_eval_smooth_part_and_gradient(B, d, lambda, x)
% fun_eval_smooth_part_and_gradient
% Evaluate the smooth part and its gradient:
%   g(x) = lambda * ||B*x - d||_2^2,
%   grad g(x) = 2 * lambda * B' * (B*x - d).

    residual = B * x - d;
    g_val = lambda * (residual' * residual);
    grad_g = 2 * lambda * (B' * residual);
end
