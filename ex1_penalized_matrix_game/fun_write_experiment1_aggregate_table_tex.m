function fun_write_experiment1_aggregate_table_tex(aggregate_rows, output_name)
% fun_write_experiment1_aggregate_table_tex
% Write the final Experiment 1 aggregate table as LaTeX code.

    if nargin < 2 || isempty(output_name)
        output_name = 'experiment1_aggregate_table.tex';
    end

    fid = fopen(output_name, 'w');
    if fid < 0
        error('Could not open %s for writing.', output_name);
    end

    cleaner = onCleanup(@() fclose(fid));
    n_rows = size(aggregate_rows, 1);

    fprintf(fid, '\\begin{table}[htbp]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{Experiment 1: averaged performance over repeated random instances. Each entry reports mean (standard deviation).}\n');
    fprintf(fid, '\\label{tab:exp1_matrix_game_repeats}\n');
    fprintf(fid, '\\small\n');
    fprintf(fid, '\\setlength{\\tabcolsep}{4pt}\n');
    fprintf(fid, '\\renewcommand{\\arraystretch}{1.1}\n');
    fprintf(fid, '\\begin{tabular}{c c c c c c c c c}\n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, '$\\varepsilon$ & FGM time & FGM gap & FGM fsubg & FGM ggrad & PFUGS time & PFUGS gap & PFUGS fsubg & PFUGS ggrad \\\\\n');
    fprintf(fid, '\\hline\n');

    for i = 1:n_rows
        fprintf(fid, '$%s$ & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n', ...
            aggregate_rows{i, 1}, aggregate_rows{i, 2}, aggregate_rows{i, 3}, ...
            aggregate_rows{i, 4}, aggregate_rows{i, 5}, aggregate_rows{i, 6}, ...
            aggregate_rows{i, 7}, aggregate_rows{i, 8}, aggregate_rows{i, 9});
    end

    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');
end
