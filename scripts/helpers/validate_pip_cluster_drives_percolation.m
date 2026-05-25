% validate_pip_cluster_drives_percolation.m
% -------------------------------------------------------------------------
% Validation that the 5 PiP-cluster nodes drive percolation of the group-
% average giant-75 binary graph.
%
% For each attack strategy we run a FULL attack (all N nodes) and record
% the size of the second-largest connected component (S2) at every step.
% The percolation point is the step at which S2 is maximised. Strategies:
%   - PiP        : full PiP rank order from AVG_giant75_PiP_2d_label_order_matlab.csv
%   - Degree     : top-down by binary degree
%   - Betweenness: top-down by binary betweenness centrality
%   - PageRank   : top-down by PageRank (alpha=0.85)
%
% BCT orderings are randomised within ties; we report mean percolation
% point over n_iter random tie-breaks but use a single representative
% trajectory (random seed fixed) for plotting/prefix analysis.
%
% Validation pieces:
%   1) Print percolation point per strategy.
%   2) Show the prefix of nodes removed up to & including the percolation
%      step for every strategy, marking which are in the 5-node PiP cluster.
%   3) Run a 5-node cluster-only attack (cluster nodes ordered by AVG PiP
%      rank within the cluster) and report whether percolation occurs by
%      step 5, plus the S2 trajectory.
%
% Outputs (under results/avg_pip_cluster_validation/):
%   - percolation_points.csv             (strategy, perc_point, n_cluster_in_prefix)
%   - attack_prefix_nodes.csv            (strategy, rank, node_matlab_1based, in_cluster, label)
%   - cluster_only_trajectory.csv        (step, S2_over_N, LCC_over_N)
%   - s2_trajectories.mat                (S2 curves for each strategy)
%   - perc_point_bar.png                 (bar of percolation point per strategy)
%   - s2_trajectories.png                (S2/N curves with percolation point marked)
%   - cluster_overlap_bar.png            (cluster-node count within each prefix)
%
% Requires: BCT on path. Set BCT_PATH env var, or edit bctPath below.
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

%% --- Inputs ---
avgAdjMat = fullfile(repoRoot, 'data', 'PSI_broadband_MEG_mats', 'avg', ...
    'AVG_broadband_psi_adj_giant75_nonexcluded.mat');
avgPipOrderCsv = fullfile(repoRoot, 'results', 'pip_cluster', ...
    'AVG_giant75_PiP_2d_label_order_matlab.csv');
clusterCsv = fullfile(repoRoot, 'figures', 'pip_cluster', 'avg_giant75', ...
    'AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW_top_cluster_nodes.csv');
labelsCsv = fullfile(repoRoot, 'data', 'MNI_66_AAL_onelinestructure.csv');

if ~isfile(avgAdjMat);      error("Missing avg adjacency MAT:\n%s", avgAdjMat);      end
if ~isfile(avgPipOrderCsv); error("Missing AVG PiP order CSV:\n%s", avgPipOrderCsv); end
if ~isfile(clusterCsv);     error("Missing cluster top-nodes CSV:\n%s", clusterCsv); end

%% --- Tunables ---
n_iter = 500;     % BCT random tie-break iterations
rng(12345);       % representative trajectory uses a fixed seed

%% --- Load adjacency (binary undirected) ---
D = load(avgAdjMat, 'psi_adj');
adj = double(D.psi_adj > 0);
adj = triu(adj, 1) + triu(adj, 1)';
n_nodes = size(adj, 1);
adj(1:n_nodes+1:end) = 0;

%% --- Load AVG PiP order ---
T_pip = readtable(avgPipOrderCsv);
if ~ismember('node_matlab', T_pip.Properties.VariableNames)
    error("Expected 'node_matlab' column in %s", avgPipOrderCsv);
end
pip_order = T_pip.node_matlab(:)';   % 1-based, descending importance
if numel(pip_order) ~= n_nodes
    error("PiP order length (%d) != n_nodes (%d).", numel(pip_order), n_nodes);
end
rank_pos = zeros(1, n_nodes);
rank_pos(pip_order) = 1:n_nodes;

%% --- Load cluster nodes ---
cluster_nodes_0based = unique(readmatrix(clusterCsv));
cluster_nodes = cluster_nodes_0based(:)' + 1;   % 1-based
cluster_nodes = cluster_nodes(cluster_nodes >= 1 & cluster_nodes <= n_nodes);
top_n = numel(cluster_nodes);
fprintf('PiP cluster nodes (1-based): %s\n', mat2str(cluster_nodes));
fprintf('Cluster size = %d\n', top_n);

%% --- Optional AAL labels for printing ---
labels = strings(n_nodes, 1);
if isfile(labelsCsv)
    raw = readlines(labelsCsv);
    if numel(raw) >= n_nodes
        labels(:) = raw(1:n_nodes);
    end
end

%% --- BCT centralities (computed once) ---
deg = degrees_und(adj);                  deg = deg(:)';
bc  = betweenness_bin(adj);              bc  = bc(:)';
pr  = pagerank_centrality(adj, 0.85);    pr  = pr(:)';

%% --- Strategy definitions ---
strategies = {'PiP','Degree','Betweenness','PageRank'};
nS = numel(strategies);
metrics = struct('Degree', deg, 'Betweenness', bc, 'PageRank', pr);

%% --- Full-attack percolation: per-strategy mean and a representative trajectory ---
perc_mean = zeros(nS, 1);
perc_repr = zeros(nS, 1);              % from the representative seeded run
s2_traj   = zeros(nS, n_nodes);        % S2/N trajectory (length N, before each removal i)
attack_repr = zeros(nS, n_nodes);

% PiP (deterministic)
[s2_traj(1,:), lcc_pip, attack_repr(1,:)] = run_full_attack(adj, pip_order);
perc_repr(1) = perc_point_from_s2(s2_traj(1,:));
perc_mean(1) = perc_repr(1);

% BCT strategies
for k = 2:nS
    metric = metrics.(strategies{k});
    perc_vals = zeros(n_iter, 1);
    for i = 1:n_iter
        ord = break_ties_randomly(metric);
        s2 = run_full_attack(adj, ord);
        perc_vals(i) = perc_point_from_s2(s2);
    end
    perc_mean(k) = mean(perc_vals);

    % representative seeded trajectory (used for prefix listing & plots)
    rng(12345);
    ord_repr = break_ties_randomly(metric);
    [s2_traj(k,:), ~, attack_repr(k,:)] = run_full_attack(adj, ord_repr);
    perc_repr(k) = perc_point_from_s2(s2_traj(k,:));
end

%% --- Cluster overlap within each strategy's prefix ---
prefix_len = ceil(perc_repr);                  % integer step count
cluster_overlap = zeros(nS, 1);
prefixes = cell(nS, 1);
for k = 1:nS
    L = max(1, prefix_len(k));
    prefixes{k} = attack_repr(k, 1:L);
    cluster_overlap(k) = sum(ismember(prefixes{k}, cluster_nodes));
end

%% --- Cluster-only attack: 5 cluster nodes ordered by AVG PiP rank ---
[~, si]            = sort(rank_pos(cluster_nodes), 'ascend');
cluster_pip_order  = cluster_nodes(si);

% Run an attack that goes cluster_pip_order then PiP order on remaining nodes,
% but record S2/LCC for the FIRST top_n steps so we can answer
% "does percolation occur by step top_n if we attack only the cluster?"
tail_order = setdiff(pip_order, cluster_pip_order, 'stable');
clust_first_full = [cluster_pip_order(:)' tail_order(:)'];
[s2_clust_full, lcc_clust_full, ~] = run_full_attack(adj, clust_first_full);

s2_clust_first_topN  = s2_clust_full(1:top_n);
lcc_clust_first_topN = lcc_clust_full(1:top_n);
perc_clust_first     = perc_point_from_s2(s2_clust_full);   % may exceed top_n

%% --- Save outputs ---
outDir = fullfile(repoRoot, 'results', 'avg_pip_cluster_validation');
if ~isfolder(outDir); mkdir(outDir); end

% percolation points + cluster overlap
fid = fopen(fullfile(outDir, 'percolation_points.csv'), 'w');
fprintf(fid, 'strategy,percolation_point_mean,percolation_point_repr,n_cluster_in_prefix,cluster_size\n');
for k = 1:nS
    fprintf(fid, '%s,%.6g,%d,%d,%d\n', strategies{k}, perc_mean(k), prefix_len(k), ...
        cluster_overlap(k), top_n);
end
fprintf(fid, 'ClusterOnly_PiPRank,%d,%d,%d,%d\n', perc_clust_first, ...
    min(perc_clust_first, top_n), top_n, top_n);
fclose(fid);

% prefix node list
fid = fopen(fullfile(outDir, 'attack_prefix_nodes.csv'), 'w');
fprintf(fid, 'strategy,rank,node_matlab_1based,node_python_0based,in_cluster,aal_label\n');
for k = 1:nS
    pre = prefixes{k};
    for r = 1:numel(pre)
        node1 = pre(r);
        in_c  = ismember(node1, cluster_nodes);
        lab   = "";
        if numel(labels) >= node1; lab = labels(node1); end
        fprintf(fid, '%s,%d,%d,%d,%d,"%s"\n', strategies{k}, r, node1, node1-1, in_c, lab);
    end
end
% cluster-only prefix (always exactly top_n entries)
for r = 1:top_n
    node1 = cluster_pip_order(r);
    lab   = "";
    if numel(labels) >= node1; lab = labels(node1); end
    fprintf(fid, '%s,%d,%d,%d,%d,"%s"\n', 'ClusterOnly_PiPRank', r, node1, node1-1, 1, lab);
end
fclose(fid);

% cluster-only trajectory
fid = fopen(fullfile(outDir, 'cluster_only_trajectory.csv'), 'w');
fprintf(fid, 'step,S2_over_N,LCC_over_N\n');
for r = 1:top_n
    fprintf(fid, '%d,%.6g,%.6g\n', r, s2_clust_first_topN(r), lcc_clust_first_topN(r));
end
fclose(fid);

% trajectories MAT
save(fullfile(outDir, 's2_trajectories.mat'), ...
    'strategies','s2_traj','attack_repr','perc_mean','perc_repr', ...
    'cluster_nodes','cluster_pip_order','s2_clust_first_topN','lcc_clust_first_topN', ...
    'perc_clust_first','top_n','n_iter','avgAdjMat','avgPipOrderCsv','clusterCsv');

%% --- Plots ---
% Percolation point bar
fig = figure('Color','w','Visible','off');
bar([perc_mean; perc_clust_first]);
set(gca, 'XTickLabel', [strategies, {'Cluster only'}]);
ylabel('Percolation point (number of nodes)');
title('Average graph: percolation point per attack strategy');
grid on; box on;
exportgraphics(fig, fullfile(outDir, 'perc_point_bar.png'), 'Resolution', 200);
close(fig);

% Cluster overlap bar
fig = figure('Color','w','Visible','off');
bar(cluster_overlap);
set(gca, 'XTickLabel', strategies);
ylabel(sprintf('# cluster nodes (out of %d) within attack prefix', top_n));
ylim([0 top_n + 0.5]);
title('Cluster nodes removed by each strategy''s percolation point');
grid on; box on;
exportgraphics(fig, fullfile(outDir, 'cluster_overlap_bar.png'), 'Resolution', 200);
close(fig);

% S2 trajectories with percolation points
fig = figure('Color','w','Visible','off');
hold on;
cmap = lines(nS);
for k = 1:nS
    plot(1:n_nodes, s2_traj(k,:), '-', 'Color', cmap(k,:), 'LineWidth', 1.4, ...
        'DisplayName', strategies{k});
    xline(perc_repr(k), '--', 'Color', cmap(k,:), 'HandleVisibility','off');
end
plot(1:top_n, s2_clust_first_topN, 'k-o', 'LineWidth', 1.4, ...
    'DisplayName', 'Cluster-only (5 nodes)');
xlabel('Removal step'); ylabel('S2 / N');
title('Second-largest component trajectory under each attack');
legend('Location','best'); grid on; box on; xlim([1 n_nodes]);
exportgraphics(fig, fullfile(outDir, 's2_trajectories.png'), 'Resolution', 200);
close(fig);

%% --- Console summary ---
fprintf('\n=== Percolation point summary (avg giant-75 graph) ===\n');
fprintf('Strategy        mean_perc_point   repr_perc_point   cluster_in_prefix\n');
for k = 1:nS
    fprintf('%-15s   %14.4g   %14d   %d / %d\n', strategies{k}, perc_mean(k), prefix_len(k), ...
        cluster_overlap(k), top_n);
end
fprintf('Cluster-only attack (PiP-rank-ordered): percolation at step %d (cluster size %d)\n', ...
    perc_clust_first, top_n);
fprintf('Outputs written to %s\n', outDir);

%% --- Helper functions ---
function idx = break_ties_randomly(metric)
    [~, sort_idx] = sort(metric, 'descend');
    if all(diff(metric(sort_idx)) ~= 0)
        idx = sort_idx;
        return
    end
    idx = [];
    uv = sort(unique(metric), 'ascend');
    for vi = numel(uv):-1:1
        nodes = find(metric == uv(vi));
        nodes = nodes(randperm(numel(nodes)));
        idx = [idx, nodes]; %#ok<AGROW>
    end
end

function [s2_traj, lcc_traj, attack_seq] = run_full_attack(adj, attack_order)
    n = size(adj, 1);
    s2_traj = zeros(1, n);
    lcc_traj = zeros(1, n);
    attack_seq = attack_order(:)';
    A = adj;
    for i = 1:n
        [~, comp_sizes] = get_components(A);
        cs = sort(comp_sizes, 'descend');
        lcc_traj(i) = cs(1) / n;
        if numel(cs) > 1
            s2_traj(i) = cs(2) / n;
        else
            s2_traj(i) = 0;
        end
        node = attack_seq(i);
        A(node, :) = 0;
        A(:, node) = 0;
    end
end

function p = perc_point_from_s2(s2_traj)
    if max(s2_traj) > (1 / numel(s2_traj))
        p = find(s2_traj == max(s2_traj), 1);
    else
        p = numel(s2_traj);
    end
end
