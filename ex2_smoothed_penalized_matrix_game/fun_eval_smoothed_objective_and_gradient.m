function [phi, grad] = fun_eval_smoothed_objective_and_gradient(A, B, d, lambda, mu, x)
% fun_eval_smoothed_objective_and_gradient
% Evaluate the smoothed objective and its gradient.

    [g_val, grad_g] = fun_eval_smooth_part_and_gradient(B, d, lambda, x);
    [f_val, grad_f] = fun_eval_smoothed_max_part(A, x, mu);
    phi = g_val + f_val;
    grad = grad_g + grad_f;
end
