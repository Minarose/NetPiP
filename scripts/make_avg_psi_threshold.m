% Build average PSI matrix across subjects and apply proportional threshold.
clear; clc;

PIP_ROOT = getenv('PIP_ROOT');
if isempty(PIP_ROOT)
    error('PIP_ROOT environment variable not set.');
end

data_dir = PIP_ROOT;
files = dir(fullfile(data_dir, '*_broadband_psi_adj.mat'));
if isempty(files)
    error('No *_broadband_psi_adj.mat files found in %s', data_dir);
end

n = 0;
sumA = [];
file_names = strings(numel(files), 1);

for i = 1:numel(files)
    fpath = fullfile(data_dir, files(i).name);
    S = load(fpath);
    if isfield(S, 'psi_adj')
        A = S.psi_adj;
    elseif isfield(S, 'A')
        A = S.A;
    else
        warning('Skipping %s (psi_adj or A not found).', files(i).name);
        continue;
    end
    A = full(A);
    if isempty(sumA)
        sumA = zeros(size(A));
    end
    sumA = sumA + A;
    n = n + 1;
    file_names(n) = string(files(i).name);
end

if n == 0
    error('No valid matrices loaded for averaging.');
end

avg_psi_adj = sumA ./ n;
avg_psi_adj(1:size(avg_psi_adj,1)+1:end) = 0; % zero diagonal

threshold_prop = 0.10;
psi_adj = threshold_proportional(avg_psi_adj, threshold_prop);

avg_dir = fullfile(PIP_ROOT, 'avg');
if ~exist(avg_dir, 'dir')
    mkdir(avg_dir);
end

out_file = fullfile(avg_dir, 'AVG_broadband_psi_adj_top10.mat');
save(out_file, 'psi_adj', 'avg_psi_adj', 'threshold_prop', 'file_names', '-v7.3');

fprintf('Saved thresholded average PSI to %s\n', out_file);

% ---------------- helper ----------------
function W_thr = threshold_proportional(W, p)
    if p <= 0 || p > 1
        error('p must be in (0,1].');
    end
    W = full(W);
    W(1:size(W,1)+1:end) = 0;
    iu = triu(true(size(W)), 1);
    w = W(iu);
    [~, idx] = sort(abs(w), 'descend');
    n_keep = max(1, round(p * numel(w)));
    keep_mask = false(size(w));
    keep_mask(idx(1:n_keep)) = true;
    w_thr = zeros(size(w));
    w_thr(keep_mask) = w(keep_mask);
    W_thr = zeros(size(W));
    W_thr(iu) = w_thr;
    W_thr = W_thr + W_thr';
end
