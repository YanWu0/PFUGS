function hit = fun_extract_threshold_report(res, epsilon_value)
% fun_extract_threshold_report
% Extract the first hitting statistics for gap <= epsilon_value.

    idx = find(res.gap_bar <= epsilon_value + res.ineq_tol, 1, 'first');
    if isempty(idx)
        idx = res.k_done;
    end

    hit = struct();
    hit.iter = idx;
    hit.time_sec = res.time_hist_sec(idx);
    hit.gap = res.gap_bar(idx);

    if isfield(res, 'oracle_hist')
        hit.f_eval = res.oracle_hist.f_eval(idx);
        hit.f_subgrad = res.oracle_hist.f_subgrad(idx);
        hit.g_eval = res.oracle_hist.g_eval(idx);
        hit.g_grad = res.oracle_hist.g_grad(idx);
    else
        hit.f_eval = NaN;
        hit.f_subgrad = NaN;
        hit.g_eval = NaN;
        hit.g_grad = NaN;
    end
end
