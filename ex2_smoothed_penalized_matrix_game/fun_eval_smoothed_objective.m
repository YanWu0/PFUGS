function phi = fun_eval_smoothed_objective(A, B, d, lambda, mu, x)
% fun_eval_smoothed_objective
% Evaluate Phi(x) = lambda * ||B*x-d||_2^2 + mu*log(sum_j exp(<A_j,x>/mu)).

    f_val = fun_eval_smoothed_max_part(A, x, mu);
    phi = lambda * norm(B * x - d)^2 + f_val;
end
