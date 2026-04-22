function phi = fun_eval_objective(A, B, d, lambda, x)
% fun_eval_objective
% Evaluate
%   Phi(x) = lambda * ||B*x - d||_2^2 + max_j <A_j, x>

    phi = lambda * norm(B * x - d)^2 + max(A' * x);
end
