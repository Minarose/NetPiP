% threshold_psi_giant75_per_subject.m
% -------------------------------------------------------------------------
% Per-subject giant-component (75%) binarization, matching the idea used in
% make_avg_psi_giantcomp_nonexcluded.py (1000 thresholds, most permissive
% threshold that still yields giant >= round(0.75*n)).
%
% Edit DATA_DIR and OUT_DIR for your machine (Windows paths OK).
% Input files: *_broadband_psi_adj.mat with variable psi_adj (dense).
% Output: one MAT per subject with BINARY psi_adj (0/1), zero diagonal,
%         undirected (explicitly symmetrized before thresholding).
% -------------------------------------------------------------------------

%% ========= USER PATHS (edit) =========
REPO_ROOT = '/Users/minaroseismail/Desktop/NetPiP-1';  % or 'C:\...\NetPiP-1'
DATA_DIR  = fullfile(REPO_ROOT, 'data', 'PSI_broadband_MEG_mats');
OUT_DIR   = fullfile(REPO_ROOT, 'data', 'PSI_broadband_MEG_mats', 'per_subject_giant75');
% =====================================

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

files = dir(fullfile(DATA_DIR, '*_broadband_psi_adj.mat'));
if isempty(files)
    error('No *_broadband_psi_adj.mat in:\n%s', DATA_DIR);
end

gcc_fraction = 0.75;
n_thresh       = 1000;

for fi = 1:numel(files)
    fp = fullfile(files(fi).folder, files(fi).name);
    S  = load(fp, 'psi_adj');
    if ~isfield(S, 'psi_adj')
        error('File missing psi_adj: %s', fp);
    end
    W = double(S.psi_adj);

    % Undirected: symmetrize, zero diagonal (same as typical PSI adjacency prep)
    W = (W + W') / 2;
    W(1:size(W,1)+1:end) = 0;

    mn = min(W(:));
    mx = max(W(:));
    if ~(isfinite(mn) && isfinite(mx)) || mx <= 0
        warning('Skipping %s (non-finite or non-positive range)', files(fi).name);
        continue;
    end

    thresholds = linspace(mn, mx, n_thresh);
    n          = size(W, 1);
    gcc_target = round(gcc_fraction * n);

    giant_ok = false(n_thresh, 1);
    for ti = 1:n_thresh
        B = (W > thresholds(ti));
        B = (B + B') / 2 > 0;  % logical symmetrize binary
        B(1:n+1:end) = 0;
        giant_ok(ti) = giant_component_order(B) >= gcc_target;
    end

    % Most permissive valid threshold = last true when scanning low -> high
    idx = find(giant_ok, 1, 'last');
    if isempty(idx)
        warning('No threshold met giant for %s — skipping', files(fi).name);
        continue;
    end

    thr_val            = thresholds(idx);
    adj_psi_percolating = double(W > thr_val);
    adj_psi_percolating = (adj_psi_percolating + adj_psi_percolating') / 2 > 0;
    adj_psi_percolating = double(adj_psi_percolating);
    adj_psi_percolating(1:n+1:end) = 0;

    dens = density_undirected(adj_psi_percolating);

    % Base name: AD01_broadband_psi_adj -> AD01_broadband_psi_adj_giant75.mat
    [~, base, ~] = fileparts(files(fi).name);
    out_name = [base, '_giant75.mat'];
    out_fp   = fullfile(OUT_DIR, out_name);

    psi_adj            = adj_psi_percolating;
    threshold_value    = thr_val;
    threshold_rule     = sprintf('giant_component_%.2f', gcc_fraction);
    gcc_target_saved   = gcc_target;
    density            = dens;
    source_file        = files(fi).name;

    save(out_fp, 'psi_adj', 'threshold_value', 'threshold_rule', ...
         'gcc_target_saved', 'density', 'source_file', '-v7.3');

    fprintf('Wrote %s  thr=%.6g  density=%.6g\n', out_fp, thr_val, dens);
end

function g = giant_component_order(B)
    % Largest connected component size for undirected binary B.
    G = graph(B, 'OmitSelfLoops');
    bins = conncomp(G);
    if isempty(bins)
        g = 0;
        return;
    end
    g = max(accumarray(bins(:), 1));
end

function d = density_undirected(B)
    n = size(B, 1);
    if n <= 1
        d = 0;
        return;
    end
    U = triu(B, 1);
    d = full(sum(U(:))) / (n * (n - 1) / 2);
end
