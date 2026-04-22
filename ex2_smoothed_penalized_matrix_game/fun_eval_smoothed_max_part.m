function [f_val, grad_f, prob] = fun_eval_smoothed_max_part(A, x, mu)
% fun_eval_smoothed_max_part
% Evaluate the log-sum-exp smoothing of max_j <A_j, x> and its gradient.

    scores = full(A' * x);
    shifted = (scores - max(scores)) / mu;
    exp_shifted = exp(shifted);
    prob = exp_shifted / sum(exp_shifted);
    f_val = max(scores) + mu * log(sum(exp_shifted));
    grad_f = A * prob;
end
