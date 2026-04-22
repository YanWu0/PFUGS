function gamma = fun_compute_gamma(Lk, Gamma_prev)
% fun_compute_gamma
% Solve
%   Lk * gamma^2 = (1-gamma) * Gamma_prev
% for gamma in (0,1].

    disc = Gamma_prev^2 + 4 * Lk * Gamma_prev;
    gamma = (-Gamma_prev + sqrt(disc)) / (2 * Lk);
end
