function oracle = fun_init_upfugs_oracle_counts()
% fun_init_upfugs_oracle_counts
% Initialize oracle counters for UPFUGS.

    oracle = struct();
    oracle.g_eval = 0;
    oracle.g_grad = 0;
    oracle.f_eval = 0;
    oracle.f_subgrad = 0;
end
