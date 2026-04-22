function x_new = fun_entropy_prox_update(x_prev, g, eta)
% fun_entropy_prox_update
% Solve
%   min_{x in Delta_n} <g,x> + eta * sum_i x_i log(x_i / x_prev_i)
% in closed form.

    logw = log(max(x_prev, realmin)) - g / eta;
    c = max(logw); 
    w = exp(logw - c); % minus c is just for numerical stability.
    x_new = w / sum(w);
end
