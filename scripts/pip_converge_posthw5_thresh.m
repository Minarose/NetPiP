% ================================================================
% pip_converge_posthw5_thresh.m
% ================================================================
% Run PiP convergence with proportional thresholding applied to psi_adj.
% Threshold proportion is controlled by THRESH_PROP (default 0.05).
% Results are saved to results_converge_5pct by default.
% ================================================================

clear; clc;
t0_total = tic;

%% ===================== ENVIRONMENT ==========================

PIP_ROOT = getenv('PIP_ROOT');
if isempty(PIP_ROOT)
    error('PIP_ROOT environment variable not set.');
end

SUBJECT_FILE = getenv('SUBJECT_FILE');
if isempty(SUBJECT_FILE)
    error('SUBJECT_FILE environment variable not set.');
end

MAX_ATTACKS = str2double(getenv('MAX_ATTACKS'));
if isnan(MAX_ATTACKS) || MAX_ATTACKS <= 0
    MAX_ATTACKS = 1e7;
end

CHUNK_SIZE = str2double(getenv('CHUNK_SIZE'));
if isnan(CHUNK_SIZE) || CHUNK_SIZE <= 0
    CHUNK_SIZE = 10000;
end

plateau_start = str2double(getenv('HW95_TOL'));
if isnan(plateau_start) || plateau_start <= 0
    plateau_start = 0.05;
end

needStable = str2double(getenv('REQUIRE_STABLE'));
if isnan(needStable) || needStable < 1
    needStable = 3;
end

range_tol = str2double(getenv('HW_RANGE_TOL'));
if isnan(range_tol) || range_tol <= 0
    range_tol = 0.005;
end

slope_tol = str2double(getenv('HW_SLOPE_TOL'));
if isnan(slope_tol) || slope_tol <= 0
    slope_tol = 1e-7;
end

win_len = str2double(getenv('PLATEAU_WINDOW'));
if isnan(win_len) || win_len < 3
    win_len = 5;
end

BASE_SEED = str2double(getenv('BASE_SEED'));
if isnan(BASE_SEED), BASE_SEED = 12345; end

THRESH_PROP = str2double(getenv('THRESH_PROP'));
if isnan(THRESH_PROP) || THRESH_PROP <= 0 || THRESH_PROP > 1
    THRESH_PROP = 0.05;
end

OUT_DIR = getenv('OUT_DIR');
if isempty(OUT_DIR)
    OUT_DIR = fullfile(PIP_ROOT, 'results_converge_5pct');
end

fprintf('\n>>> SUBJECT=%s\n', SUBJECT_FILE);
fprintf('    MAX_ATTACKS=%d, CHUNK=%d\n', MAX_ATTACKS, CHUNK_SIZE);
fprintf('    plateau_start=%.3f, range_tol=%.4f, slope_tol=%.1e, win_len=%d, needStable=%d\n', ...
    plateau_start, range_tol, slope_tol, win_len, needStable);
fprintf('    THRESH_PROP=%.2f\n\n', THRESH_PROP);

%% ===================== LOAD + THRESHOLD =====================

data_dir = PIP_ROOT;
S = load(fullfile(data_dir, SUBJECT_FILE));

if isfield(S, 'psi_adj')
    A = S.psi_adj;
elseif isfield(S, 'A')
    A = S.A;
else
    error('Subject %s did not contain variable psi_adj or A.', SUBJECT_FILE);
end

A = threshold_proportional(A, THRESH_PROP);
A = sparse(A);

nNodes = size(A,1);

%% ===================== PARPOOL ==============================

slurm_cpus = getenv('SLURM_CPUS_PER_TASK');
if isempty(slurm_cpus)
    nWorkers = feature('numCores');
else
    nWorkers = max(1, str2double(slurm_cpus));
end

p = gcp('nocreate');
if isempty(p) || p.NumWorkers ~= nWorkers
    parpool('local', nWorkers);
end

maxNumCompThreads(1);

%% ===================== RNG SETUP ============================

task_id = getenv('SLURM_ARRAY_TASK_ID');
if isempty(task_id)
    task_id = getenv('SLURM_JOB_ID');
end
if isempty(task_id)
    task_id = '0';
end

rng(BASE_SEED + str2double(task_id), 'Threefry');

%% ===================== ACCUMULATORS ==========================

counts_per_step = zeros(nNodes, 1, 'uint64');
part_counts     = zeros(nNodes, nNodes, 'uint64');

stableCount = 0;
n_attacks   = 0;

attacks_hist = [];
hw95_hist    = [];

%% ===================== MAIN LOOP ============================

while n_attacks < MAX_ATTACKS
    this_chunk = min(CHUNK_SIZE, MAX_ATTACKS - n_attacks);
    if this_chunk <= 0
        break;
    end

    fprintf('--- chunk: current=%d  adding=%d\n', n_attacks, this_chunk);

    psteps   = zeros(this_chunk, 1, 'uint32');
    hits_mat = zeros(this_chunk, nNodes, 'uint16');

    tchunk = tic;
    parfor a = 1:this_chunk
        atk = randperm(nNodes);
        mask = true(1, nNodes);
        second_comp = zeros(1, nNodes-1, 'uint32');

        for step = 1:nNodes-1
            mask(atk(step)) = false;
            sub_adj = A(mask, mask);

            if nnz(sub_adj) == 0
                bin2 = 0;
            else
                G  = graph(sub_adj);
                cc = conncomp(G);
                K  = max(cc);
                if K < 2
                    bin2 = 0;
                else
                    h = histcounts(cc, 1:(double(K)+1));
                    h = sort(h, 'descend');
                    bin2 = h(2);
                end
            end
            second_comp(step) = uint32(bin2);
        end

        smax = max(second_comp);
        if smax > 1
            psteps(a) = uint32(find(second_comp == smax, 1, 'first'));
        else
            psteps(a) = uint32(nNodes);
        end

        idx = atk(1:psteps(a));
        row = zeros(1, nNodes, 'uint16');
        row(idx) = 1;
        hits_mat(a,:) = row;
    end
    fprintf('    ✓ parfor done in %.2f min\n', toc(tchunk)/60);

    n_attacks = n_attacks + this_chunk;
    counts_per_step = counts_per_step + uint64(accumarray(double(psteps), 1, [nNodes,1]));

    for a = 1:this_chunk
        ps = double(psteps(a));
        part_counts(ps,:) = part_counts(ps,:) + uint64(hits_mat(a,:));
    end

    node_P = zeros(nNodes, nNodes, 'double');
    for p = 1:nNodes
        c = double(counts_per_step(p));
        if c > 0
            node_P(p,:) = double(part_counts(p,:)) / c;
        else
            node_P(p,:) = NaN;
        end
    end

    hw_mat = wilson_hw_matrix(part_counts, counts_per_step);
    hw95 = prctile(hw_mat(:), 95);

    attacks_hist(end+1,1) = n_attacks; %#ok<AGROW>
    hw95_hist(end+1,1)    = hw95;      %#ok<AGROW>

    [is_plateau, slope_val, range_val, mean_val] = ...
        check_hw_plateau(attacks_hist, hw95_hist, plateau_start, range_tol, slope_tol, win_len);

    if is_plateau
        stableCount = stableCount + 1;
    else
        stableCount = 0;
    end

    fprintf('    hw95=%.3f  mean=%.3f  range=%.4f  slope=%.2e  plateau=%d  stable=%d\n', ...
            hw95, mean_val, range_val, slope_val, is_plateau, stableCount);

    if stableCount >= needStable
        fprintf('*** Plateau convergence reached at %d attacks. Stopping early. ***\n', n_attacks);
        break;
    end
end

elapsed_total = toc(t0_total);

%% ===================== SAVE RESULTS ==========================

if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

[~, base] = fileparts(SUBJECT_FILE);
out_mat = fullfile(OUT_DIR, sprintf('%s_ConvHW.mat', base));

meta = struct();
meta.subject_file   = SUBJECT_FILE;
meta.MAX_ATTACKS    = MAX_ATTACKS;
meta.CHUNK_SIZE     = CHUNK_SIZE;
meta.nNodes         = nNodes;
meta.elapsed_sec    = elapsed_total;
meta.attacks_hist   = attacks_hist;
meta.hw95_hist      = hw95_hist;
meta.plateau_start  = plateau_start;
meta.range_tol      = range_tol;
meta.slope_tol      = slope_tol;
meta.win_len        = win_len;
meta.needStable     = needStable;
meta.THRESH_PROP    = THRESH_PROP;
meta.OUT_DIR        = OUT_DIR;

save(out_mat, 'node_P', 'counts_per_step', 'part_counts', 'meta', '-v7.3');

fprintf('\nSaved converged PiP → %s\n', out_mat);

%% ===================== HELPERS ==============================

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

function hw_mat = wilson_hw_matrix(part_counts, counts_per_step)
z = 1.96;
[nDepths, nNodes] = size(part_counts);
hw_mat = zeros(nDepths, nNodes);

for d = 1:nDepths
    n = double(counts_per_step(d));
    if n == 0
        hw_mat(d, :) = NaN;
        continue;
    end

    k = double(part_counts(d,:));
    p = k ./ n;
    p(isnan(p)) = 0;

    z2  = z^2;
    num = z .* sqrt((p .* (1-p) ./ n) + (z2 ./ (4*n^2)));
    den = 1 + (z2 / n);
    hw_mat(d,:) = num ./ den;
end
end

function [is_plateau, slope_val, range_val, mean_val] = ...
    check_hw_plateau(attacks_hist, hw95_hist, plateau_start, range_tol, slope_tol, win_len)
n = numel(hw95_hist);
if n < win_len
    is_plateau = false;
    slope_val  = NaN;
    range_val  = NaN;
    mean_val   = NaN;
    return;
end

hw = hw95_hist(end-win_len+1:end);
atk = attacks_hist(end-win_len+1:end);

mean_val  = mean(hw);
range_val = max(hw) - min(hw);
slope_val = polyfit(atk, hw, 1);
slope_val = slope_val(1);

is_plateau = (mean_val < plateau_start) && (range_val < range_tol) && (abs(slope_val) < slope_tol);
end
