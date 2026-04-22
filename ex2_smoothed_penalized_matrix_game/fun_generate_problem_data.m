function [A, B, d, info] = fun_generate_problem_data(params)
% fun_generate_problem_data
% Generate A, B, d for the penalized matrix-game benchmark.

    n = params.n;
    m = params.m;
    p = params.p;

    % -------- A generation --------
    switch lower(params.A_distribution)
        case 'uniform_pm1'
            if params.A_density >= 1
                A = 2 * rand(n, m) - 1;
            else
                A_mask = rand(n, m) < params.A_density;
                A = (2 * rand(n, m) - 1) .* A_mask;
                A = sparse(A);
            end
        case 'normal'
            if params.A_density >= 1
                A = randn(n, m);
            else
                A_mask = rand(n, m) < params.A_density;
                A = randn(n, m) .* A_mask;
                A = sparse(A);
            end
        otherwise
            error('Unknown A_distribution: %s', params.A_distribution);
    end

    A = params.A_scale * A;

    % -------- B generation --------
    switch lower(params.B_distribution)
        case 'normal'
            if params.B_density >= 1
                B_raw = randn(p, n);
            else
                B_mask = rand(p, n) < params.B_density;
                B_raw = randn(p, n) .* B_mask;
            end
        case 'uniform_pm1'
            if params.B_density >= 1
                B_raw = 2 * rand(p, n) - 1;
            else
                B_mask = rand(p, n) < params.B_density;
                B_raw = (2 * rand(p, n) - 1) .* B_mask;
            end
        otherwise
            error('Unknown B_distribution: %s', params.B_distribution);
    end

    B = params.B_scale * B_raw;

    % -------- d generation --------
    switch lower(params.d_distribution)
        case 'normal'
            d = randn(p, 1);
        case 'uniform_pm1'
            d = 2 * rand(p, 1) - 1;
        otherwise
            error('Unknown d_distribution: %s', params.d_distribution);
    end
    d = params.B_scale * d;

    % -------- Diagnostics --------
    info = struct();
    info.cost_ratio_nnz = nnz(B) / max(nnz(A), 1);
    info.L_f = full(max(max(abs(A))));
    info.A_scale = params.A_scale;
    info.L_q_euclid = 2 * params.lambda * norm(B, 2)^2;
    info.B_scale = params.B_scale;
    info.lambda = params.lambda;
    info.d_strategy = 'scale_with_B';
end
