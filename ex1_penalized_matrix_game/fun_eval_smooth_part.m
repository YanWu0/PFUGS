function g_val = fun_eval_smooth_part(B, d, lambda, x)
% fun_eval_smooth_part
% Evaluate the smooth part
%   g(x) = lambda * ||B*x - d||_2^2.

    residual = B * x - d;
    g_val = lambda * (residual' * residual);
end
