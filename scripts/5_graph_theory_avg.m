% 5_graph_theory_avg.m
% -------------------------------------------------------------------------
% Compute percolation point on ONE average binary graph using:
%   Degree, Betweenness, PageRank (BCT) + PiP order derived from AVG node_P.
%
% This is the "average analysis" analogue of 5_graph_theory_per_subject.m,
% but for a single cohort-average adjacency (one matrix).
%
% Inputs (defaults are the canonical repo locations; edit if needed):
%   - avgAdjMat: AVG_* binary adjacency MAT containing variable `psi_adj`
%   - avgConvMat: AVG_*_ConvHW.mat containing variable `node_P` (PiP matrix)
%
% Output:
%   - Figures: bar chart of percolation point for the four strategies.
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

%% --- Inputs (edit if your files are elsewhere) ---
avgAdjMat = fullfile(repoRoot, 'data', 'PSI_broadband_MEG_mats', 'group_average', ...
    'AVG_broadband_psi_adj_giant75_nonexcluded.mat');
avgConvMat = fullfile(repoRoot, 'results', 'pip_convergence', 'avg_giant75', ...
    'AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat');

if ~isfile(avgAdjMat)
    error("Missing avg adjacency MAT:\n%s\nSync/copy it into that path or edit avgAdjMat.", avgAdjMat);
end
if ~isfile(avgConvMat)
    error("Missing avg PiP ConvHW MAT:\n%s\nSync/copy it into that path or edit avgConvMat.", avgConvMat);
end

%% --- Load adjacency (binary undirected) ---
D = load(avgAdjMat, 'psi_adj');
if ~isfield(D, 'psi_adj')
    error("avgAdjMat missing variable `psi_adj`: %s", avgAdjMat);
end
adj = double(D.psi_adj > 0);
adj = triu(adj, 1) + triu(adj, 1)';
n_nodes = size(adj, 1);
adj(1:n_nodes+1:end) = 0;

%% --- PiP order from node_P (tilt τ=S/6, clip negative to 0) ---
S = load(avgConvMat, 'node_P');
if ~isfield(S, 'node_P')
    error("avgConvMat missing variable `node_P`: %s", avgConvMat);
end
pip_order = pip_order_from_nodeP(S.node_P); % MATLAB 1-based indices
if numel(pip_order) ~= n_nodes
    error("PiP order length (%d) does not match adj nodes (%d).", numel(pip_order), n_nodes);
end

%% --- Percolation points ---
n_iter = 500; % only affects tie-breaking for BCT metric orders

% Degree
deg = degrees_und(adj);
deg = deg(:)';
vals = zeros(n_iter, 1);
for i = 1:n_iter
    ord = break_ties_randomly(deg);
    vals(i) = run_attack_once(adj, ord);
end
perc_deg = mean(vals);

% Betweenness
bc = betweenness_bin(adj);
bc = bc(:)';
vals = zeros(n_iter, 1);
for i = 1:n_iter
    ord = break_ties_randomly(bc);
    vals(i) = run_attack_once(adj, ord);
end
perc_bc = mean(vals);

% PageRank
pr = pagerank_centrality(adj, 0.85);
pr = pr(:)';
vals = zeros(n_iter, 1);
for i = 1:n_iter
    ord = break_ties_randomly(pr);
    vals(i) = run_attack_once(adj, ord);
end
perc_pr = mean(vals);

% PiP (single pass)
perc_pip = run_attack_once(adj, pip_order(:)');

% Cluster-first (A/1): cluster nodes first (ordered by PiP order), then the rest in PiP order
clusterCsv = fullfile(repoRoot, 'results', 'pip_cluster', 'avg_giant75_top_cluster_nodes_labeled.csv');
if ~isfile(clusterCsv)
    error("Missing cluster CSV:\n%s\nRun step 4 first (scripts/4_cluster_pip_set.py + scripts/helpers/label_cluster_nodes.py).", clusterCsv);
end
% Prefer preserving original headers; also tolerate MATLAB renaming.
try
    T = readtable(clusterCsv, 'VariableNamingRule', 'preserve');
catch
    % Older MATLAB: ignore
    T = readtable(clusterCsv);
end

vnames = string(T.Properties.VariableNames);

% Choose the column that looks most like 0-based indices (0..n_nodes-1).
best_col = "";
best_vals = [];
best_score = -inf;
for ci = 1:numel(vnames)
    raw = T.(vnames(ci));
    if isnumeric(raw)
        vals = double(raw(:));
    elseif iscell(raw)
        vals = str2double(string(raw(:)));
    elseif isstring(raw)
        vals = str2double(raw(:));
    elseif iscategorical(raw)
        vals = str2double(string(raw(:)));
    else
        continue
    end
    vals = vals(isfinite(vals));
    if isempty(vals)
        continue
    end
    inrange = (vals >= 0) & (vals <= (n_nodes - 1));
    score = sum(inrange);
    % Prefer a column whose name contains 0based if tie.
    if score > best_score || (score == best_score && contains(lower(vnames(ci)), "0based"))
        best_score = score;
        best_col = vnames(ci);
        best_vals = vals(inrange);
    end
end
if strlength(best_col) == 0 || isempty(best_vals)
    error("Could not find a numeric 0-based index column in %s", clusterCsv);
end

cluster_nodes = unique(best_vals + 1); % MATLAB 1-based nodes
cluster_nodes = cluster_nodes(cluster_nodes >= 1 & cluster_nodes <= n_nodes);
if isempty(cluster_nodes)
    error("No valid cluster nodes loaded from %s", clusterCsv);
end

% Build rank position in PiP order: rank(node) = position in pip_order (1..N)
rank = zeros(1, n_nodes);
rank(pip_order) = 1:n_nodes;
[~, si] = sort(rank(cluster_nodes), 'ascend');
cluster_sorted = cluster_nodes(si);
tail = setdiff(pip_order, cluster_sorted, 'stable');
cluster_first_order = [cluster_sorted(:)' tail(:)'];
perc_cluster_pip = run_attack_once(adj, cluster_first_order);

%% --- Plot ---
figure('Color', 'w');
bar([perc_deg, perc_bc, perc_pr, perc_pip, perc_cluster_pip]);
xticklabels({'Degree', 'Betweenness', 'PageRank', 'PiP', 'Cluster→PiP'});
ylabel('Percolation point');
title('Percolation point on average graph (single matrix)');
grid on; box on;

%% --- Save numeric results (for downstream plots) ---
outDir = fullfile(repoRoot, 'results', 'graph_theory_overlap');
if ~isfolder(outDir)
    mkdir(outDir);
end
outCsv = fullfile(outDir, 'avg_percolation_points_matlab.csv');
fid = fopen(outCsv, 'w');
fprintf(fid, 'metric,percolation_point\n');
fprintf(fid, 'Degree,%.12g\n', perc_deg);
fprintf(fid, 'Betweenness,%.12g\n', perc_bc);
fprintf(fid, 'PageRank,%.12g\n', perc_pr);
fprintf(fid, 'PiP,%.12g\n', perc_pip);
fprintf(fid, 'Cluster_to_PiP,%.12g\n', perc_cluster_pip);
fclose(fid);
save(fullfile(outDir, 'avg_percolation_points_matlab.mat'), ...
    'perc_deg', 'perc_bc', 'perc_pr', 'perc_pip', 'perc_cluster_pip', ...
    'avgAdjMat', 'avgConvMat', 'clusterCsv', 'cluster_sorted', 'cluster_first_order');
fprintf('Saved percolation points to %s\n', outCsv);

%% --- Helper functions ---
function idx = break_ties_randomly(metric)
    % Sort desc; within ties, randomize.
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
    % Percolation point defined as first i maximizing size of 2nd-largest component.
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

function order_matlab = pip_order_from_nodeP(node_P)
    % Derive node removal order used for the τ=S/6 tilted 2D heatmap.
    % - non-finite -> 0
    % - negative clipped to 0
    % - tilt: multiply row s by exp(-s/tau), tau=max(2,S/6)
    % - rank nodes by largest tilted peak amplitude (desc), tie-break by earlier peak step
    P = node_P;
    P = crop_longest_non_nan_block(P);
    if isempty(P)
        error("Empty node_P after crop.");
    end
    P(~isfinite(P)) = 0;
    P(P < 0) = 0;
    S = size(P, 1);
    tau = max(2, S / 6);
    s = (1:S)';
    w = exp(-s ./ tau);
    Pt = P .* w; % implicit expand over columns

    [peak_amp, peak_idx] = max(Pt, [], 1);
    peak_idx = peak_idx(:);
    peak_amp = peak_amp(:);

    % Sort: primary -peak_amp (desc), secondary peak_idx (asc)
    [~, order0] = sortrows([-peak_amp, peak_idx], [1 2]); %#ok<ASGLU>
    order_matlab = order0(:)'; % 1..N, MATLAB 1-based
end

function Pc = crop_longest_non_nan_block(P)
    % Keep the longest contiguous block of rows that has at least one non-NaN value.
    if isempty(P)
        Pc = P;
        return
    end
    row_ok = any(~isnan(P), 2);
    if ~any(row_ok)
        Pc = [];
        return
    end
    d = diff([false; row_ok; false]);
    starts = find(d == 1);
    ends   = find(d == -1) - 1;
    lens   = ends - starts + 1;
    [~, imax] = max(lens);
    Pc = P(starts(imax):ends(imax), :);
end

