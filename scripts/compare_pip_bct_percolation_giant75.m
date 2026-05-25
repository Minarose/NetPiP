% compare_pip_bct_percolation_giant75.m
% -------------------------------------------------------------------------
% Percolation-point comparison on per-subject giant-75% binary graphs:
%   Degree, Betweenness, PageRank (random tie-break, mean over n_iter)
%   vs PiP removal order from giant75_per_subject_pip2d_order_matlab_noheader.csv
% Figures: per-subject bars, group mean+-SEM, density scatter, violin (KDE) of
% per-subject percolation index across the four metrics.
%
% Open from repo: in MATLAB, cd to NetPiP-1/scripts and run this file, or
% add NetPiP-1/scripts to path and run by name.
%
% Requires: Brain Connectivity Toolbox on path. Set BCT_PATH env var, or
% edit bctPath below (default: ~/Downloads/BCT/2019_03_03_BCT).
% -------------------------------------------------------------------------

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);

%% --- BCT ---
bctPath = getenv('BCT_PATH');
if isempty(bctPath)
    bctPath = fullfile(getenv('HOME'), 'Downloads', 'BCT', '2019_03_03_BCT');
end
if ~isfolder(bctPath)
    error(['BCT folder not found: %s\nSet env BCT_PATH or edit bctPath in ', mfilename], bctPath);
end
addpath(bctPath);

%% --- Data (relative to repo root) ---
matDir = fullfile(repoRoot, 'data', 'PSI_broadband_MEG_mats', 'per_subject_giant75_nonexcluded');
pipCsv = fullfile(repoRoot, 'data', 'PSI_broadband_MEG_mats', 'results_converge_giant75_per_subject', ...
    'giant75_per_subject_pip2d_order_matlab_noheader.csv');

if ~isfile(pipCsv)
    error('Missing PiP order CSV:\n%s\nRun export_pip2d_label_order_matlab.py batch mode first.', pipCsv);
end

pip_orders = readmatrix(pipCsv);

files = dir(fullfile(matDir, '*_giant75.mat'));
if isempty(files)
    error('No *_giant75.mat in:\n%s', matDir);
end
[~, ix] = sort({files.name});
files = files(ix);

n_subjects = numel(files);
n_nodes    = size(pip_orders, 2);
assert(n_subjects == size(pip_orders, 1), ...
    'Row mismatch: %d MAT files vs %d CSV rows (check sorting vs export).', n_subjects, size(pip_orders,1));

n_iter = 500;

perc_deg = zeros(n_subjects, 1);
perc_bc  = zeros(n_subjects, 1);
perc_pr  = zeros(n_subjects, 1);
perc_pip = zeros(n_subjects, 1);
dens     = zeros(n_subjects, 1);

for s = 1:n_subjects
    data = load(fullfile(files(s).folder, files(s).name), 'psi_adj');
    adj  = data.psi_adj;

    adj = double(adj > 0);
    adj = triu(adj, 1) + triu(adj, 1)';
    adj(1:n_nodes+1:end) = 0;

    m       = nnz(triu(adj, 1));
    dens(s) = m / (n_nodes * (n_nodes - 1) / 2);

    % Degree
    deg = degrees_und(adj);
    deg = deg(:)';
    perc_vals = zeros(n_iter, 1);
    for i = 1:n_iter
        deg_order   = break_ties_randomly(deg);
        perc_vals(i) = run_attack_once(adj, deg_order);
    end
    perc_deg(s) = mean(perc_vals);

    % Betweenness
    bc = betweenness_bin(adj);
    bc = bc(:)';
    perc_vals = zeros(n_iter, 1);
    for i = 1:n_iter
        bc_order      = break_ties_randomly(bc);
        perc_vals(i) = run_attack_once(adj, bc_order);
    end
    perc_bc(s) = mean(perc_vals);

    % PageRank
    pr = pagerank_centrality(adj, 0.85);
    pr = pr(:)';
    perc_vals = zeros(n_iter, 1);
    for i = 1:n_iter
        pr_order      = break_ties_randomly(pr);
        perc_vals(i) = run_attack_once(adj, pr_order);
    end
    perc_pr(s) = mean(perc_vals);

    % PiP order (single pass); row s matches sorted *_giant75.mat
    pip_order    = pip_orders(s, :);
    perc_pip(s) = run_attack_once(adj, pip_order);

    fprintf('Subject %d/%d %s\n', s, n_subjects, files(s).name);
end

%% --- Per-subject barplot ---
figure('Color', 'w');
bar([perc_deg, perc_bc, perc_pr, perc_pip]);
legend({'Degree', 'Betweenness', 'PageRank', 'PiP'}, 'Location', 'northwest');
xlabel('Subject (sorted ID)'); ylabel('Percolation point');
title('Percolation point (giant-75 graphs)');

%% --- Group mean +- SEM ---
mean_vals = mean([perc_deg, perc_bc, perc_pr, perc_pip], 1);
sem_vals  = std([perc_deg, perc_bc, perc_pr, perc_pip], 0, 1) ./ sqrt(n_subjects);

figure('Color', 'w');
bar(mean_vals); hold on;
errorbar(1:4, mean_vals, sem_vals, 'k.', 'LineWidth', 1.5);
xlabel('Attack strategy'); ylabel('Mean percolation point');
title('Group-level comparison (giant-75)');
xticklabels({'Degree', 'Betweenness', 'PageRank', 'PiP'});

%% --- Density vs percolation ---
metrics = {'Degree', 'Betweenness', 'PageRank', 'PiP'};
Y       = [perc_deg, perc_bc, perc_pr, perc_pip];

figure('Color', 'w');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for k = 1:4
    nexttile;
    x = dens;
    y = Y(:, k);
    scatter(x, y, 36, 'filled'); hold on; grid on;
    p    = polyfit(x, y, 1);
    xfit = linspace(min(x), max(x), 100);
    yfit = polyval(p, xfit);
    plot(xfit, yfit, 'LineWidth', 1.5);
    [r, pval] = corr(x, y, 'Type', 'Pearson', 'Rows', 'complete');
    xlim_curr = xlim;
    ylim_curr = ylim;
    tx = xlim_curr(1) + 0.03 * range(xlim_curr);
    ty = ylim_curr(2) - 0.08 * range(ylim_curr);
    text(tx, ty, sprintf('slope = %.3g, r = %.2f, p = %.3g', p(1), r, pval), ...
        'FontSize', 9, 'VerticalAlignment', 'top');
    xlabel('Matrix density (binary)');
    ylabel('Percolation point');
    title(sprintf('%s vs. density', metrics{k}));
end

%% --- Violin: per-subject percolation point by metric (nodes until collapse) ---
% KDE mirror (ksdensity) + patch; jittered subject points. Y-axis = attack step
% index at which the second-largest component peaks (same as bar/scatter above).
figure('Color', 'w'); hold on;
labelsV = {'Degree', 'Betweenness', 'PageRank', 'PiP'};
Yv     = [perc_deg, perc_bc, perc_pr, perc_pip];
nM     = 4;
cV     = [0.20 0.45 0.85; 0.90 0.45 0.15; 0.35 0.65 0.35; 0.55 0.25 0.65];
halfW  = 0.38;
for k = 1:nM
    d = Yv(:, k);
    d = d(~isnan(d));
    if isempty(d)
        continue
    end
    if numel(d) >= 2 && license('test', 'Statistics_Toolbox')
        [f, xi] = ksdensity(d, 'NumPoints', 512);
    else
        % Fallback: Gaussian KDE with fixed bandwidth (no Statistics Toolbox)
        sig = max(std(d), (prctile(d, 75) - prctile(d, 25)) / 1.34);
        sig = max(sig, range(d) * 0.05 + eps);
        xi = linspace(min(d), max(d), 256);
        f = zeros(size(xi));
        for ii = 1:numel(xi)
            f(ii) = mean(exp(-0.5 * ((d - xi(ii)) / sig) .^ 2)) / (sig * sqrt(2 * pi));
        end
    end
    if max(f) < eps
        f = ones(size(f));
    end
    f = f / max(f) * halfW;
    pos = k;
    xpatch = [pos + f, fliplr(pos - f)];
    ypatch = [xi, fliplr(xi)];
    fill(xpatch, ypatch, cV(k, :), 'FaceAlpha', 0.35, 'EdgeColor', cV(k, :), 'LineWidth', 1);
    md = median(d);
    plot([pos - halfW * 0.9, pos + halfW * 0.9], [md, md], 'k-', 'LineWidth', 1.8);
    jx = pos + (rand(numel(d), 1) - 0.5) * 0.09;
    scatter(jx, d, 28, 'k', 'filled', 'MarkerFaceAlpha', 0.45);
end
xlim([0.4, nM + 0.6]);
set(gca, 'XTick', 1:nM, 'XTickLabel', labelsV);
ylabel('Percolation point (node removal index)');
title('Percolation point across subjects: four metrics (giant-75)');
grid on; box on;
hold off;

%% --- Helper functions ---
function idx = break_ties_randomly(metric)
    [~, sort_idx] = sort(metric, 'descend');
    tied = find(diff(metric(sort_idx)) == 0);
    if isempty(tied)
        idx = sort_idx;
    else
        idx = [];
        uv = sort(unique(metric), 'ascend');
        for vi = numel(uv):-1:1
            nodes = find(metric == uv(vi));
            nodes = nodes(randperm(numel(nodes)));
            idx = [idx, nodes]; %#ok<AGROW>
        end
    end
end

function perc_point = run_attack_once(adj, attack_order)
    n = size(adj, 1);
    second_comp = zeros(1, n);
    adj_tmp = adj;
    for i = 1:n
        [~, comp_sizes] = get_components(adj_tmp);
        tmp = sort(comp_sizes);
        if numel(tmp) > 1
            second_comp(i) = tmp(end - 1);
        else
            second_comp(i) = 0;
        end
        node = attack_order(i);
        adj_tmp(node, :) = 0;
        adj_tmp(:, node) = 0;
    end
    if max(second_comp) > 1
        perc_point = find(second_comp == max(second_comp), 1);
    else
        perc_point = n;
    end
end
