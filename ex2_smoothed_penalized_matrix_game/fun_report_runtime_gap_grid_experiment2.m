function fun_report_runtime_gap_grid_experiment2(res_upfgs, ags_base_results, ...
    ags_inner_guess_list, ags_outer_guess_list, max_runtime_sec, output_dir, show_figures, plot_runtime_sec)
% fun_report_runtime_gap_grid_experiment2
% Create gap-vs-time grid PDFs. Each subplot compares PFUGS with one AGS
% run using a fixed (M_guess, L_guess) pair.

    if nargin < 6 || isempty(output_dir)
        output_dir = pwd;
    end
    if nargin < 7 || isempty(show_figures)
        show_figures = false;
    end
    if nargin < 8 || isempty(plot_runtime_sec)
        plot_runtime_sec = max_runtime_sec;
    end

    n_inner = numel(ags_inner_guess_list);
    n_outer = numel(ags_outer_guess_list);
    n_plots = n_inner * n_outer;
    n_cols = 3;
    n_rows = ceil(n_plots / n_cols);

    upfgs_time = [0; res_upfgs.time_hist_sec(:)];
    upfgs_gap = [res_upfgs.gap0; res_upfgs.gap_bar(:)];
    epsilon_value = res_upfgs.epsilon;

    fig = local_build_gap_grid_figure(n_rows, n_cols, upfgs_time, upfgs_gap, ...
        ags_base_results, ags_inner_guess_list, ags_outer_guess_list, ...
        plot_runtime_sec, show_figures, true, true, epsilon_value);
    exportgraphics(fig, fullfile(output_dir, 'figure_experiment2_gap_grid_nref_epsilon.pdf'), ...
        'ContentType', 'vector');
    local_finish_figure(fig, show_figures);
end

function fig = local_build_gap_grid_figure(n_rows, n_cols, upfgs_time, upfgs_gap, ...
    ags_base_results, ags_inner_guess_list, ags_outer_guess_list, plot_runtime_sec, ...
    show_figures, truncate_ags_at_nref, show_epsilon_line, epsilon_value)

    if show_figures
        fig = figure('Color', 'w', 'Visible', 'on');
    else
        fig = figure('Color', 'w', 'Visible', 'off');
    end
    set(fig, 'Units', 'normalized', 'Position', [0.01, 0.08, 0.98, min(0.86, 0.25 * n_rows + 0.10)]);
    tl = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_idx = 0;
    n_inner = numel(ags_inner_guess_list);
    n_outer = numel(ags_outer_guess_list);
    n_plots = n_inner * n_outer;
    M_reference = ags_inner_guess_list(ceil(n_inner / 2));
    L_reference = ags_outer_guess_list(ceil(n_outer / 2));
    for i = 1:n_inner
        for j = 1:n_outer
            plot_idx = plot_idx + 1;
            res_ags = ags_base_results{i, j};
            [upfgs_plot_time, upfgs_plot_gap] = local_clip_curve( ...
                upfgs_time, upfgs_gap, plot_runtime_sec);
            [ags_time, ags_gap, ags_stops_at_nref, ags_nref_time] = local_get_ags_plot_curve( ...
                res_ags, truncate_ags_at_nref);
            [ags_time, ags_gap] = local_clip_curve(ags_time, ags_gap, plot_runtime_sec);

            ax = nexttile(tl, plot_idx);
            ax.Toolbar.Visible = 'off';
            semilogy(ax, upfgs_plot_time, max(upfgs_plot_gap, realmin), 'k--', 'LineWidth', 2.4);
            hold(ax, 'on');
            semilogy(ax, ags_time, max(ags_gap, realmin), 'k-', 'LineWidth', 1.1);
            if show_epsilon_line
                semilogy(ax, [0, plot_runtime_sec], epsilon_value * [1, 1], ...
                    'k:', 'LineWidth', 1.4);
            end
            hold(ax, 'off');

            xlim(ax, [0, plot_runtime_sec]);
            local_set_gap_ylim(ax, upfgs_plot_gap, ags_gap, show_epsilon_line, epsilon_value);
            local_set_base2_yticks(ax, show_epsilon_line, epsilon_value);

            ax.TickLabelInterpreter = 'latex';
            ax.FontSize = 21;
            ax.FontWeight = 'bold';
            ax.LineWidth = 1.8;
            local_add_time_unit_xtick(ax, plot_runtime_sec);
            grid(ax, 'on');
            xlabel(ax, local_panel_label(plot_idx, res_ags, ...
                ags_inner_guess_list(i), ags_outer_guess_list(j), M_reference, ...
                L_reference, ags_stops_at_nref, ags_nref_time, plot_runtime_sec), ...
                'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');

            legend_entries = {'PFUGS', 'AGS'};
            if plot_idx < n_plots
                legend_location = 'northeast';
            else
                legend_location = local_choose_right_legend_location(ax, upfgs_plot_time, ...
                    upfgs_plot_gap, ags_time, ags_gap);
            end
            legend(ax, legend_entries, 'Location', legend_location, ...
                'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
end

function legend_location = local_choose_right_legend_location(ax, upfgs_time, upfgs_gap, ags_time, ags_gap)
    x_limits = xlim(ax);
    y_limits = ylim(ax);
    x_cut = x_limits(1) + 0.58 * (x_limits(2) - x_limits(1));
    log_y_limits = log10(y_limits);
    log_y_cut = log_y_limits(1) + 0.62 * (log_y_limits(2) - log_y_limits(1));

    top_right_is_busy = local_curve_hits_region(upfgs_time, upfgs_gap, x_cut, log_y_cut) ...
        || local_curve_hits_region(ags_time, ags_gap, x_cut, log_y_cut);

    if top_right_is_busy
        legend_location = 'east';
    else
        legend_location = 'northeast';
    end
end

function is_busy = local_curve_hits_region(time_values, gap_values, x_cut, log_y_cut)
    valid_idx = isfinite(time_values) & isfinite(gap_values) & gap_values > 0;
    if ~any(valid_idx)
        is_busy = false;
        return;
    end

    time_values = time_values(valid_idx);
    gap_values = gap_values(valid_idx);
    is_busy = any(time_values >= x_cut & log10(gap_values) >= log_y_cut);
end

function [ags_time, ags_gap, ags_stops_at_nref, ags_nref_time] = local_get_ags_plot_curve(res_ags, truncate_ags_at_nref)
    if truncate_ags_at_nref && isfield(res_ags, 'N_ref')
        cut_k = min(res_ags.k_done, res_ags.N_ref);
    else
        cut_k = res_ags.k_done;
    end
    ags_stops_at_nref = truncate_ags_at_nref && isfield(res_ags, 'N_ref') ...
        && res_ags.k_done >= res_ags.N_ref && cut_k == res_ags.N_ref;
    if ags_stops_at_nref
        ags_nref_time = res_ags.time_hist_sec(res_ags.N_ref);
    else
        ags_nref_time = NaN;
    end

    ags_time = [0; res_ags.time_hist_sec(1:cut_k)];
    ags_gap = [res_ags.gap0; res_ags.gap_bar(1:cut_k)];
end

function [plot_time, plot_gap] = local_clip_curve(time_values, gap_values, plot_runtime_sec)
    time_values = time_values(:);
    gap_values = gap_values(:);
    keep_idx = time_values <= plot_runtime_sec;
    plot_time = time_values(keep_idx);
    plot_gap = gap_values(keep_idx);

    if isempty(plot_time)
        plot_time = 0;
        plot_gap = gap_values(1);
    end

    first_after_idx = find(time_values > plot_runtime_sec, 1, 'first');
    if ~isempty(first_after_idx) && plot_time(end) < plot_runtime_sec
        t_left = plot_time(end);
        g_left = plot_gap(end);
        t_right = time_values(first_after_idx);
        g_right = gap_values(first_after_idx);
        theta = (plot_runtime_sec - t_left) / max(t_right - t_left, realmin);
        g_interp = local_log_linear_gap_interp(g_left, g_right, theta);
        plot_time = [plot_time; plot_runtime_sec];
        plot_gap = [plot_gap; g_interp];
    end
end

function g_interp = local_log_linear_gap_interp(g_left, g_right, theta)
    if isfinite(g_left) && isfinite(g_right) && g_left > 0 && g_right > 0
        g_interp = 10^(log10(g_left) + theta * (log10(g_right) - log10(g_left)));
    else
        g_interp = g_left + theta * (g_right - g_left);
    end
end

function local_finish_figure(fig, show_figures)
    if show_figures
        figure(fig);
        shg;
        drawnow;
    else
        close(fig);
    end
end

function label = local_panel_label(plot_idx, res_ags, M_guess, L_guess, M_reference, ...
    L_reference, ags_stops_at_nref, ags_nref_time, plot_runtime_sec)
    M_label = local_format_relative_guess(M_guess, M_reference, 'M_{\tilde f}');
    L_label = local_format_relative_guess(L_guess, L_reference, 'L_g');
    if ags_stops_at_nref && ags_nref_time <= plot_runtime_sec
        stop_label = 'stop at $N$';
    else
        stop_label = local_format_display_stop_reason(res_ags, plot_runtime_sec);
    end
    label = sprintf('(%d) AGS: $\\widehat M_{\\tilde f}=%s$, $\\widehat L_g=%s$, %s', ...
        plot_idx, M_label, L_label, stop_label);
end

function label = local_format_relative_guess(value, reference_value, base_symbol)
    ratio = value / reference_value;
    if abs(ratio - 1) <= 1e-10
        label = base_symbol;
    elseif abs(ratio - 100) <= 1e-10
        label = sprintf('100%s', base_symbol);
    elseif abs(ratio - 0.01) <= 1e-10
        label = sprintf('%s/100', base_symbol);
    elseif ratio > 1
        label = sprintf('%s%s', local_format_multiplier(ratio), base_symbol);
    else
        label = sprintf('%s/%s', base_symbol, local_format_multiplier(1 / ratio));
    end
end

function label = local_format_multiplier(value)
    if abs(value - round(value)) <= 1e-10
        label = sprintf('%.0f', value);
    else
        label = local_format_constant(value);
    end
end

function label = local_format_display_stop_reason(res_ags, plot_runtime_sec)
    epsilon_hit_idx = find(res_ags.gap_bar <= res_ags.epsilon + res_ags.ineq_tol, 1, 'first');
    if ~isempty(epsilon_hit_idx) && res_ags.time_hist_sec(epsilon_hit_idx) <= plot_runtime_sec
        label = 'stop due to $\epsilon$';
        return;
    end

    switch lower(res_ags.stop_reason)
        case 'epsilon'
            label = local_not_stopped_label(plot_runtime_sec);
        case 'time_limit'
            if res_ags.runtime_sec <= plot_runtime_sec
                label = 'stop due to time limit';
            else
                label = local_not_stopped_label(plot_runtime_sec);
            end
        case 'maxit'
            if ~isempty(res_ags.time_hist_sec) && res_ags.time_hist_sec(end) <= plot_runtime_sec
                label = 'stop due to maxit';
            else
                label = local_not_stopped_label(plot_runtime_sec);
            end
        otherwise
            label = local_not_stopped_label(plot_runtime_sec);
    end
end

function label = local_not_stopped_label(plot_runtime_sec)
    label = sprintf('not stopped by %.1g sec', plot_runtime_sec);
end

function local_add_time_unit_xtick(ax, plot_runtime_sec)
    current_ticks = xticks(ax);
    tolerance = 1e-10 * max(1, abs(plot_runtime_sec));
    current_ticks = current_ticks(current_ticks >= 0 & current_ticks <= plot_runtime_sec);
    if ~any(abs(current_ticks - plot_runtime_sec) <= tolerance)
        current_ticks = sort([current_ticks(:); plot_runtime_sec]);
    end

    tick_labels = cell(size(current_ticks));
    for idx = 1:numel(current_ticks)
        if abs(current_ticks(idx) - plot_runtime_sec) <= tolerance
            tick_labels{idx} = sprintf('$%g\\,\\mathrm{sec}$', plot_runtime_sec);
        else
            tick_labels{idx} = sprintf('$%g$', current_ticks(idx));
        end
    end

    xticks(ax, current_ticks);
    xticklabels(ax, tick_labels);
end

function label = local_format_constant(value)
    if value == 0
        label = '0';
    elseif abs(value) >= 1e3 || abs(value) < 1e-2
        exponent_value = round(log10(abs(value)));
        mantissa_value = value / 10^exponent_value;
        if abs(mantissa_value - round(mantissa_value)) < 1e-10
            mantissa_text = sprintf('%.0f', mantissa_value);
        else
            mantissa_text = sprintf('%.2g', mantissa_value);
        end
        if abs(mantissa_value - 1) < 1e-10
            label = sprintf('10^{%d}', exponent_value);
        else
            label = sprintf('%s\\cdot 10^{%d}', mantissa_text, exponent_value);
        end
    elseif abs(value - round(value)) < 1e-10
        label = sprintf('%.0f', value);
    else
        label = sprintf('%.3g', value);
    end
end

function local_set_gap_ylim(ax, upfgs_gap, ags_gap, show_epsilon_line, epsilon_value)
    all_gap = [upfgs_gap(:); ags_gap(:)];
    if show_epsilon_line
        all_gap = [all_gap; epsilon_value];
    end
    all_gap = all_gap(isfinite(all_gap) & all_gap > 0);
    if isempty(all_gap)
        return;
    end

    y_min = max(realmin, 0.8 * min(all_gap));
    y_max = max(realmin, 1.2 * max(all_gap));
    if y_max <= y_min
        y_max = 10 * y_min;
    end
    ylim(ax, [y_min, y_max]);
end

function local_set_base2_yticks(ax, show_epsilon_line, epsilon_value)
    y_limits = ylim(ax);
    if y_limits(1) <= 0 || y_limits(2) <= 0
        return;
    end

    min_exp = ceil(log2(y_limits(1)));
    max_exp = floor(log2(y_limits(2)));
    if min_exp > max_exp
        return;
    end

    candidate_exps = max_exp:-1:min_exp;
    max_tick_count = 5;
    step_size = max(1, ceil(numel(candidate_exps) / max_tick_count));
    tick_exps = candidate_exps(1:step_size:end);
    tick_values = 2 .^ tick_exps;

    tolerance = 1e-10 * max(1, epsilon_value);
    if show_epsilon_line && epsilon_value > y_limits(1) && epsilon_value < y_limits(2)
        tick_values = tick_values(abs(tick_values - epsilon_value) > tolerance);
        tick_values = [tick_values(:); epsilon_value];
    end

    tick_values = tick_values(tick_values >= y_limits(1) & tick_values <= y_limits(2));
    tick_values = sort(tick_values);
    tick_labels = cell(size(tick_values));
    for idx = 1:numel(tick_values)
        if show_epsilon_line && abs(tick_values(idx) - epsilon_value) <= tolerance
            tick_labels{idx} = local_format_epsilon_label(epsilon_value);
        else
            tick_labels{idx} = local_format_base2_tick_label(tick_values(idx));
        end
    end

    yticks(ax, tick_values);
    yticklabels(ax, tick_labels);
end

function label = local_format_epsilon_label(epsilon_value)
    log2_eps = log2(epsilon_value);
    rounded_log2_eps = round(log2_eps);
    if abs(log2_eps - rounded_log2_eps) < 1e-10
        label = sprintf('$\\epsilon=2^{%d}$', rounded_log2_eps);
    else
        label = sprintf('$\\epsilon=%.1e$', epsilon_value);
    end
end

function label = local_format_base2_tick_label(value)
    exponent_value = log2(value);
    rounded_exponent = round(exponent_value);
    if abs(exponent_value - rounded_exponent) < 1e-10
        label = sprintf('$2^{%d}$', rounded_exponent);
    else
        label = sprintf('$%.1g$', value);
    end
end
