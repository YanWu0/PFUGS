%% scr_run_fgm_matrixgame_benchmark.m
% Experiment 1: penalized matrix-game benchmark.

clear; clc; close all;

%% =========================
% parameters
% ==========================
data_params = struct();
data_params.seed = 1;
data_params.num_repeats = 10;
data_params.n = 2000;
data_params.m = 500;
data_params.p = 2000;
data_params.A_density = 1;
data_params.B_density = 1;
data_params.A_distribution = 'uniform_pm1';
data_params.B_distribution = 'uniform_pm1';
data_params.d_distribution = 'uniform_pm1';
data_params.A_scale = 1;
data_params.B_scale = 1; % scale of B and d
data_params.lambda = 0.05;

epsilon_list = 2.^(-(5));

solver = struct();
solver.display = 'off';
solver.tol_fun = 1e-9;
solver.tol_x = 1e-9;
solver.max_iter = 2000;

fgm_alg = struct();
fgm_alg.maxit = 3000000;
fgm_alg.ineq_tol = 1e-12;
fgm_alg.L0 = 1; 
fgm_alg.x0_type = 'uniform';
fgm_alg.verbose = true;
fgm_alg.print_first = 3;
fgm_alg.print_every = 5000;

pfugs_alg = struct();
pfugs_alg.maxit = fgm_alg.maxit;
pfugs_alg.max_inner_it = fgm_alg.maxit;
pfugs_alg.ineq_tol = fgm_alg.ineq_tol;
pfugs_alg.M0 = fgm_alg.L0;
pfugs_alg.x0_type = fgm_alg.x0_type;
pfugs_alg.verbose = fgm_alg.verbose;
pfugs_alg.print_first = 3;
pfugs_alg.print_every = 50;

report = struct();
report.output_dir = fileparts(mfilename('fullpath'));
report.save_results_mat = true;
report.results_mat_name = 'results_fgm_matrixgame_benchmark.mat';
report.save_latex_table = true;
report.latex_table_name = 'experiment1_aggregate_table.tex';

num_eps = numel(epsilon_list);
all_cases = cell(num_eps, data_params.num_repeats);
summary_rows_by_eps = cell(num_eps, 1);
aggregate_rows = cell(num_eps, 9);

for eps_idx = 1:num_eps
    eps_target = epsilon_list(eps_idx);
    summary_rows = zeros(data_params.num_repeats, 9);

    fprintf('\n############################################################\n');
    fprintf('Starting repeated Experiment 1 runs for epsilon = 2^{%d} (%.2e)\n', ...
        round(log2(eps_target)), eps_target);

    for repeat_idx = 1:data_params.num_repeats
        params = data_params;
        params.seed = data_params.seed + (repeat_idx - 1);

        case_tag = sprintf('eps2^%d_seed%d_n%d_m%d_p%d', ...
            round(log2(eps_target)), params.seed, params.n, params.m, params.p);

        rng(params.seed);
        [A, B, d, data_info] = fun_generate_problem_data(params);
        switch lower(fgm_alg.x0_type)
            case 'uniform'
                x0 = ones(params.n, 1) / params.n;
            case 'random_simplex'
                x0 = rand(params.n, 1);
                x0 = x0 / sum(x0);
            otherwise
                error('Unknown x0_type: %s', fgm_alg.x0_type);
        end

        fprintf('\n============================================================\n');
        fprintf('Generated penalized matrix-game instance (%s)\n', case_tag);
        fprintf('  n = %d, m = %d, p = %d\n', params.n, params.m, params.p);
        fprintf('  nnz(A)/numel(A) = %.2f\n', nnz(A)/numel(A));
        fprintf('  nnz(B)/numel(B) = %.2f\n', nnz(B)/numel(B));
        fprintf('  max_j ||A_j||_inf = %.2e\n', data_info.L_f);
        fprintf('  2*lambda*||B||_2^2 = %.2e\n', data_info.L_q_euclid);
        fprintf('  cost proxy nnz(B)/nnz(A) = %.2f\n', data_info.cost_ratio_nnz);

        sol_qp = fun_compute_optval_qp(A, B, d, params.lambda, solver);
        fprintf('\nReference solution from solver:\n');
        fprintf('  reference optimal value        = %.2e\n', sol_qp.phi_star);
        fprintf('  quadratic penalty at optimum   = %.2e\n', sol_qp.quad_term);
        fprintf('  max term at optimum            = %.2e\n', sol_qp.max_term);
        fprintf('  solver exitflag                = %d\n', sol_qp.exitflag);
        fprintf('  solver time                    = %.2f sec\n', sol_qp.runtime_sec);

        fgm_alg_case = fgm_alg;
        pfugs_alg_case = pfugs_alg;
        fgm_alg_case.epsilon = eps_target;
        pfugs_alg_case.epsilon = eps_target;

        phi_star = sol_qp.phi_star;
        fprintf('\n=== Experiment 1 (%s): trajectory to epsilon = 2^{%d} (%.2e) ===\n', ...
            case_tag, round(log2(eps_target)), eps_target);

        res_pfugs_full = fun_run_upfugs_matrixgame(A, B, d, params.lambda, pfugs_alg_case, phi_star, x0);
        res_fgm_full = fun_run_fgm_matrixgame_benchmark(A, B, d, params.lambda, fgm_alg_case, phi_star, x0);

        result_fgm = fun_extract_threshold_report(res_fgm_full, eps_target);
        result_pfugs = fun_extract_threshold_report(res_pfugs_full, eps_target);

        summary_rows(repeat_idx, :) = [ ...
            params.seed, ...
            result_fgm.time_sec, result_fgm.gap, result_fgm.f_subgrad, result_fgm.g_grad, ...
            result_pfugs.time_sec, result_pfugs.gap, result_pfugs.f_subgrad, result_pfugs.g_grad];

        if repeat_idx == 1
            fprintf('\nPer-seed summary rows for epsilon = 2^{%d}:\n', round(log2(eps_target)));
            fprintf('%-10s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
                'seed', ...
                'FGM time', 'FGM gap', 'FGM fsubg', 'FGM ggrad', ...
                'PFUGS time', 'PFUGS gap', 'PFUGS fsubg', 'PFUGS ggrad');
        end
        fprintf('%-10d %-12.2f %-12.2e %-12.2f %-12.2f %-12.2f %-12.2e %-12.2f %-12.2f\n', ...
            params.seed, ...
            result_fgm.time_sec, result_fgm.gap, ...
            result_fgm.f_subgrad, result_fgm.g_grad, ...
            result_pfugs.time_sec, result_pfugs.gap, ...
            result_pfugs.f_subgrad, result_pfugs.g_grad);

        case_summary = struct();
        case_summary.case_tag = case_tag;
        case_summary.epsilon = eps_target;
        case_summary.params = params;
        case_summary.data_info = data_info;
        case_summary.x0 = x0;
        case_summary.sol_qp = sol_qp;
        case_summary.res_fgm_full = res_fgm_full;
        case_summary.res_pfugs_full = res_pfugs_full;
        case_summary.result_fgm = result_fgm;
        case_summary.result_pfugs = result_pfugs;
        all_cases{eps_idx, repeat_idx} = case_summary;
    end

    summary_rows_by_eps{eps_idx} = summary_rows;
    metric_means = mean(summary_rows(:, 2:end), 1);
    metric_stds = std(summary_rows(:, 2:end), 0, 1);

    fprintf('\nComplete summary for epsilon = 2^{%d}:\n', round(log2(eps_target)));
    fprintf('%-10s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
        'seed', ...
        'FGM time', 'FGM gap', 'FGM fsubg', 'FGM ggrad', ...
        'PFUGS time', 'PFUGS gap', 'PFUGS fsubg', 'PFUGS ggrad');
    for repeat_idx = 1:data_params.num_repeats
        fprintf('%-10.0f %-12.2f %-12.2e %-12.2f %-12.2f %-12.2f %-12.2e %-12.2f %-12.2f\n', ...
            summary_rows(repeat_idx, 1), ...
            summary_rows(repeat_idx, 2), summary_rows(repeat_idx, 3), ...
            summary_rows(repeat_idx, 4), summary_rows(repeat_idx, 5), ...
            summary_rows(repeat_idx, 6), summary_rows(repeat_idx, 7), ...
            summary_rows(repeat_idx, 8), summary_rows(repeat_idx, 9));
    end
    fprintf('%-10s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
        'mean(std)', ...
        sprintf('%.2f(%.2f)', metric_means(1), metric_stds(1)), ...
        sprintf('%.2e(%.2e)', metric_means(2), metric_stds(2)), ...
        sprintf('%.2f(%.2f)', metric_means(3), metric_stds(3)), ...
        sprintf('%.2f(%.2f)', metric_means(4), metric_stds(4)), ...
        sprintf('%.2f(%.2f)', metric_means(5), metric_stds(5)), ...
        sprintf('%.2e(%.2e)', metric_means(6), metric_stds(6)), ...
        sprintf('%.2f(%.2f)', metric_means(7), metric_stds(7)), ...
        sprintf('%.2f(%.2f)', metric_means(8), metric_stds(8)));

    aggregate_rows(eps_idx, :) = { ...
        sprintf('2^{%d}', round(log2(eps_target))), ...
        sprintf('%.2f(%.2f)', metric_means(1), metric_stds(1)), ...
        sprintf('%.2e(%.2e)', metric_means(2), metric_stds(2)), ...
        sprintf('%.2f(%.2f)', metric_means(3), metric_stds(3)), ...
        sprintf('%.2f(%.2f)', metric_means(4), metric_stds(4)), ...
        sprintf('%.2f(%.2f)', metric_means(5), metric_stds(5)), ...
        sprintf('%.2e(%.2e)', metric_means(6), metric_stds(6)), ...
        sprintf('%.2f(%.2f)', metric_means(7), metric_stds(7)), ...
        sprintf('%.2f(%.2f)', metric_means(8), metric_stds(8))};
end

fprintf('\nAggregate summary over repeats by epsilon:\n');
fprintf('%-10s %-16s %-18s %-16s %-16s %-16s %-18s %-16s %-16s\n', ...
    'epsilon', ...
    'FGM time', 'FGM gap', 'FGM fsubg', 'FGM ggrad', ...
    'PFUGS time', 'PFUGS gap', 'PFUGS fsubg', 'PFUGS ggrad');
for eps_idx = 1:num_eps
    fprintf('%-10s %-16s %-18s %-16s %-16s %-16s %-18s %-16s %-16s\n', ...
        aggregate_rows{eps_idx, 1}, aggregate_rows{eps_idx, 2}, ...
        aggregate_rows{eps_idx, 3}, aggregate_rows{eps_idx, 4}, ...
        aggregate_rows{eps_idx, 5}, aggregate_rows{eps_idx, 6}, ...
        aggregate_rows{eps_idx, 7}, aggregate_rows{eps_idx, 8}, ...
        aggregate_rows{eps_idx, 9});
end

if report.save_latex_table
    latex_table_path = fullfile(report.output_dir, report.latex_table_name);
    fun_write_experiment1_aggregate_table_tex(aggregate_rows, latex_table_path);
    fprintf('\nSaved LaTeX aggregate table to %s\n', latex_table_path);
end

if report.save_results_mat
    summary = struct();
    summary.data_params = data_params;
    summary.fgm_alg = fgm_alg;
    summary.pfugs_alg = pfugs_alg;
    summary.report = report;
    summary.solver = solver;
    summary.epsilon_list = epsilon_list;
    summary.summary_rows_by_eps = summary_rows_by_eps;
    summary.aggregate_rows = aggregate_rows;
    summary.cases = all_cases;
    results_mat_path = fullfile(report.output_dir, report.results_mat_name);
    save(results_mat_path, 'summary');
    fprintf('\nSaved results to %s\n', results_mat_path);
end
