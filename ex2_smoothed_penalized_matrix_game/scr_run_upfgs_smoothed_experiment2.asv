 %% scr_run_upfgs_smoothed_experiment2.m
% Driver script for Experiment 2: smoothed matrix-game solved by PFUGS.

clear; clc; close all;

params = struct();
params.seed = 1;
params.n = 2000;
params.m = 500;
params.p = 2000;
params.A_density = 1;
params.B_density = 1;
params.A_distribution = 'uniform_pm1';
params.B_distribution = 'uniform_pm1';
params.d_distribution = 'uniform_pm1';
params.A_scale = 1;
params.B_scale = 1;
params.lambda = 0.05;
params.mu = 0.0001;

solver = struct();
solver.display = 'off';
solver.tol_fun = 1e-9;
solver.tol_x = 1e-9;
solver.max_iter = 5000;

alg = struct();
alg.maxit = 30000;
alg.ineq_tol = 1e-12;
alg.L0 = 1;
alg.M0 = alg.L0;
alg.x0_type = 'uniform';
alg.verbose = true;
alg.print_first = 5;
alg.print_every = 20;
alg.max_inner_it = 3000000;
alg.epsilon = 2^(-10);

ags_q = 1;
ags_guess_base = 100;

report = struct();
report.save_results_mat = false;
report.results_mat_name = 'results_upfgs_smoothed_experiment2.mat';
report.max_runtime_sec = 7;
report.plot_runtime_sec = 5;
report.output_dir = fileparts(mfilename('fullpath'));
report.show_figures = true;

cache = struct();
cache.use_saved_problem_reference = true;
cache.save_problem_reference = false;
cache.problem_reference_mat_name = 'experiment2_problem_reference.mat';
cache.problem_reference_path = fullfile(report.output_dir, cache.problem_reference_mat_name);


if cache.use_saved_problem_reference
    if ~isfile(cache.problem_reference_path)
        error('Saved problem/reference file not found: %s', cache.problem_reference_path);
    end
    loaded = load(cache.problem_reference_path, 'problem_cache');
    problem_cache = loaded.problem_cache;

    params = problem_cache.params;
    A = problem_cache.A;
    B = problem_cache.B;
    d = problem_cache.d;
    data_info = problem_cache.data_info;
    x0 = problem_cache.x0;
    L_g_entropy = problem_cache.L_g_entropy;
    M_tilde_f_entropy_ub = problem_cache.M_tilde_f_entropy_ub;
    sol_ref = problem_cache.sol_ref;
    phi_star = sol_ref.phi_star;

    fprintf('Loaded saved Experiment 2 problem/reference from %s\n', cache.problem_reference_path);
    fprintf('Experiment 2 smoothed matrix-game instance\n');
    fprintf('  n = %d, m = %d, p = %d\n', params.n, params.m, params.p);
    fprintf('  mu = %.4e\n', params.mu);
    fprintf('  nnz(A)/numel(A) = %.4f\n', nnz(A)/numel(A));
    fprintf('  nnz(B)/numel(B) = %.4f\n', nnz(B)/numel(B));
    fprintf('  exact L_g in l1-geometry         = %.12e\n', L_g_entropy);
    fprintf('  upper bound for M_{tilde f}      = %.12e\n', M_tilde_f_entropy_ub);
    fprintf('  d generation strategy = scale_with_B\n');
else
    rng(params.seed);
    [A, B, d, data_info] = fun_generate_problem_data(params);
    L_g_entropy = fun_compute_entropy_smooth_constant_g(B, params.lambda);
    M_tilde_f_entropy_ub = fun_compute_entropy_smooth_constant_smoothed_max_upper(A, params.mu);
    switch lower(alg.x0_type)
        case 'uniform'
            x0 = ones(params.n, 1) / params.n;
        case 'random_simplex'
            x0 = rand(params.n, 1);
            x0 = x0 / sum(x0);
        otherwise
            error('Unknown x0_type: %s', alg.x0_type);
    end

    fprintf('Experiment 2 smoothed matrix-game instance\n');
    fprintf('  n = %d, m = %d, p = %d\n', params.n, params.m, params.p);
    fprintf('  mu = %.4e\n', params.mu);
    fprintf('  nnz(A)/numel(A) = %.4f\n', nnz(A)/numel(A));
    fprintf('  nnz(B)/numel(B) = %.4f\n', nnz(B)/numel(B));
    fprintf('  exact L_g in l1-geometry         = %.12e\n', L_g_entropy);
    fprintf('  upper bound for M_{tilde f}      = %.12e\n', M_tilde_f_entropy_ub);
    fprintf('  d generation strategy = scale_with_B\n');

    fprintf('\n=== Solving reference problem for Experiment 2 ===\n');
    sol_ref = fun_compute_optval_smoothed_problem(A, B, d, params.lambda, params.mu, solver, x0);
    phi_star = sol_ref.phi_star;

    if cache.save_problem_reference
        problem_cache = struct();
        problem_cache.params = params;
        problem_cache.solver = solver;
        problem_cache.A = A;
        problem_cache.B = B;
        problem_cache.d = d;
        problem_cache.data_info = data_info;
        problem_cache.x0 = x0;
        problem_cache.L_g_entropy = L_g_entropy;
        problem_cache.M_tilde_f_entropy_ub = M_tilde_f_entropy_ub;
        problem_cache.sol_ref = sol_ref;
        save(cache.problem_reference_path, 'problem_cache', '-v7.3');
        fprintf('Saved Experiment 2 problem/reference to %s\n', cache.problem_reference_path);
    end
end

ags_outer_guess_list = fun_build_guess_list_around_value(L_g_entropy, ags_q, ags_guess_base);
ags_inner_guess_list = fun_build_guess_list_around_value(M_tilde_f_entropy_ub, ags_q, ags_guess_base);
fprintf('  AGS guess base                   = %.12e\n', ags_guess_base);
fprintf('  AGS outer guess range (q=%d)     = [%.12e, %.12e]\n', ...
    ags_q, ags_outer_guess_list(1), ags_outer_guess_list(end));
fprintf('  AGS inner guess range (q=%d)     = [%.12e, %.12e]\n', ...
    ags_q, ags_inner_guess_list(1), ags_inner_guess_list(end));

fprintf('  reference optimal value        = %.12e\n', sol_ref.phi_star);
fprintf('  quadratic penalty at optimum   = %.12e\n', sol_ref.quad_term);
fprintf('  smoothed max term at optimum   = %.12e\n', sol_ref.smoothed_max_term);
fprintf('  solver exitflag                = %d\n', sol_ref.exitflag);
fprintf('  solver time                    = %.12f sec\n', sol_ref.runtime_sec);

fprintf('\n=== Running PFUGS on Experiment 2 ===\n');
alg.max_runtime_sec = report.max_runtime_sec;
res_upfgs_smoothed = fun_run_upfugs_smoothed_matrixgame( ...
    A, B, d, params.lambda, params.mu, alg, phi_star, x0);

fprintf('\nExperiment 2 PFUGS summary:\n');
fprintf('  final phi(bar_x_k) = %.12e\n', res_upfgs_smoothed.phi_bar(end));
fprintf('  final gap          = %.12e\n', res_upfgs_smoothed.gap_bar(end));
fprintf('  runtime (sec)      = %.12f\n', res_upfgs_smoothed.runtime_sec);
fprintf('  stop_reason        = %s\n', res_upfgs_smoothed.stop_reason);
fprintf('  iterations used    = %d\n', res_upfgs_smoothed.k_done);
fprintf('  f evals            = %d\n', res_upfgs_smoothed.oracle.f_eval);
fprintf('  f grads            = %d\n', res_upfgs_smoothed.oracle.f_subgrad);
fprintf('  g evals            = %d\n', res_upfgs_smoothed.oracle.g_eval);
fprintf('  g grads            = %d\n', res_upfgs_smoothed.oracle.g_grad);

fprintf('\n=== Running AGS benchmark grid on Experiment 2 ===\n');
fprintf('  max runtime per run = %.2f sec\n', report.max_runtime_sec);

n_outer = numel(ags_outer_guess_list);
n_inner = numel(ags_inner_guess_list);
ags_base_results = cell(n_inner, n_outer);
fprintf('Running one AGS trajectory per (M_guess, L_guess) pair for at most %.2f seconds\n', report.max_runtime_sec);
for i = 1:n_inner
    for j = 1:n_outer
        ags_alg = struct();
        ags_alg.epsilon = alg.epsilon;
        ags_alg.ineq_tol = alg.ineq_tol;
        ags_alg.verbose = false;
        ags_alg.print_first = 5;
        ags_alg.print_every = 10000;
        ags_alg.maxit = alg.maxit;
        ags_alg.L_guess = ags_outer_guess_list(j);
        ags_alg.M_guess = ags_inner_guess_list(i);
        ags_alg.max_runtime_sec = report.max_runtime_sec;

        fprintf('  AGS pair (%d,%d): M_f_guess = %.2e, L_g_guess = %.2e ...\n', ...
            i, j, ags_alg.M_guess, ags_alg.L_guess);
        ags_base_results{i, j} = fun_run_ags_smoothed_matrixgame( ...
            A, B, d, params.lambda, params.mu, ags_alg, phi_star, x0);
        fprintf('    finished: stop = %s, time = %.2f sec, outer_it = %d, T = %d\n', ...
            ags_base_results{i, j}.stop_reason, ags_base_results{i, j}.runtime_sec, ...
            ags_base_results{i, j}.k_done, ags_base_results{i, j}.T_used);
    end
end

fprintf('\nExperiment 2 AGS benchmark summary:\n');
fprintf('%-12s %-12s %-8s %-8s %-12s %-12s %-12s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    'M_f_guess', 'L_g_guess', 'outer', 'inner', 'time', 'gap', 'stop', ...
    'outer_it', 'T', 'N_ref', 'f_eval', 'f_grad', 'g_eval', 'g_grad');
for i = 1:n_inner
    for j = 1:n_outer
        res_ags = ags_base_results{i, j};
        if isempty(res_ags.gap_bar)
            final_gap_ags = NaN;
        else
            final_gap_ags = res_ags.gap_bar(end);
        end
        fprintf('%-12.2e %-12.2e %-8s %-8s %-12.2f %-12.2e %-12s %-10d %-10d %-10d %-10d %-10d %-10d %-10d\n', ...
            res_ags.M_guess, res_ags.L_guess, res_ags.outer_component, res_ags.inner_component, ...
            res_ags.runtime_sec, final_gap_ags, res_ags.stop_reason, ...
            res_ags.k_done, res_ags.T_used, res_ags.N_ref, ...
            res_ags.oracle.f_eval, res_ags.oracle.f_subgrad, ...
            res_ags.oracle.g_eval, res_ags.oracle.g_grad);
    end
end

fun_report_runtime_gap_grid_experiment2( ...
    res_upfgs_smoothed, ags_base_results, ags_inner_guess_list, ...
    ags_outer_guess_list, report.max_runtime_sec, report.output_dir, ...
    report.show_figures, report.plot_runtime_sec);
fprintf('\nSaved Experiment 2 N-ref truncated gap-grid plot with epsilon line to %s\n', ...
    fullfile(report.output_dir, 'figure_experiment2_gap_grid_nref_epsilon.pdf'));

if report.save_results_mat
    summary = struct();
    summary.params = params;
    summary.alg = alg;
    summary.report = report;
    summary.cache = cache;
    summary.data_info = data_info;
    summary.x0 = x0;
    summary.solver = solver;
    summary.sol_ref = sol_ref;
    summary.res_upfgs_smoothed = res_upfgs_smoothed;
    summary.ags_q = ags_q;
    summary.ags_guess_base = ags_guess_base;
    summary.ags_outer_guess_list = ags_outer_guess_list;
    summary.ags_inner_guess_list = ags_inner_guess_list;
    summary.ags_base_results = ags_base_results;
    save(report.results_mat_name, 'summary');
    fprintf('\nSaved results to %s\n', report.results_mat_name);
end

fprintf('\nExperiment 2 driver finished.\n');
