function x_new = fun_entropy_two_reference_prox_update(x_anchor, x_prev, grad, eta, p)
% fun_entropy_two_reference_prox_update
% Solve
%   min_{x in Delta_n} <grad, x> + eta * V(x_anchor, x) + p * V(x_prev, x)
% in closed form for the entropy Bregman divergence on the simplex.

    coeff = eta + p;
    if coeff <= 0
        error('The combined prox coefficient eta + p must be positive.');
    end

    logw = (eta * log(max(x_anchor, realmin)) ...
        + p * log(max(x_prev, realmin)) - grad) / coeff;
    shift = max(logw);
    w = exp(logw - shift);
    x_new = w / sum(w);
end
