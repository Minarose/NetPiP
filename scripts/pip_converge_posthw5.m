% ================================================================
% pip_converge_single_subject_by2.m
% ================================================================
% Run PiP convergence for ONE subject using ONLY the
% second-component percolation rule.
%
% - Parallelized with parfor (attacks within subject)
% - Stops early when PiP converges via a plateau in hw95:
%       * hw95 < plateau_start (e.g. 0.05)
%       * small range over recent chunks
%       * near-zero slope over recent chunks
%
% Environment variables (set in SLURM script):
%   PIP_ROOT
%   SUBJECT_FILE
%   MAX_ATTACKS, CHUNK_SIZE
%   HW95_TOL          (used here as plateau_start; e.g. 0.05)
%   REQUIRE_STABLE    (how many plateau hits in a row; e.g. 3)
%   HW_RANGE_TOL      (optional, default 0.005)
%   HW_SLOPE_TOL      (optional, default 1e-7)
%   PLATEAU_WINDOW    (optional, default 5 chunks)
%   BASE_SEED
%   SLURM_CPUS_PER_TASK
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

% We now interpret HW95_TOL as the *magnitude* threshold to start
% looking for a plateau (e.g. 0.05)
plateau_start = str2double(getenv('HW95_TOL'));
if isnan(plateau_start) || plateau_start <= 0
    plateau_start = 0.05;
end

needStable = str2double(getenv('REQUIRE_STABLE'));
if isnan(needStable) || needStable < 1
    needStable = 3;
end

% Additional plateau parameters
range_tol = str2double(getenv('HW_RANGE_TOL'));
if isnan(range_tol) || range_tol <= 0
    range_tol = 0.005;   % max allowed variation in hw95 over window
end

slope_tol = str2double(getenv('HW_SLOPE_TOL'));
if isnan(slope_tol) || slope_tol <= 0
    slope_tol = 1e-7;    % near-zero slope threshold
end

win_len = str2double(getenv('PLATEAU_WINDOW'));
if isnan(win_len) || win_len < 3
    win_len = 5;         % number of chunks used to assess plateau
end

BASE_SEED = str2double(getenv('BASE_SEED'));
if isnan(BASE_SEED), BASE_SEED = 12345; end

fprintf('\n>>> SUBJECT=%s\n', SUBJECT_FILE);
fprintf('    MAX_ATTACKS=%d, CHUNK=%d\n', MAX_ATTACKS, CHUNK_SIZE);
fprintf('    plateau_start=%.3f, range_tol=%.4f, slope_tol=%.1e, win_len=%d, needStable=%d\n\n', ...
    plateau_start, range_tol, slope_tol, win_len, needStable);

%% ===================== LOAD ADJACENCY =======================

data_dir = PIP_ROOT;
S = load(fullfile(data_dir, SUBJECT_FILE));

if isfield(S, 'psi_adj')
    A = sparse(S.psi_adj);
elseif isfield(S, 'A')
    A = sparse(S.A);
else
    error('Subject %s did not contain variable psi_adj or A.', SUBJECT_FILE);
end

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

% For plotting / diagnostics
attacks_hist = [];
hw95_hist    = [];

%% ===================== MAIN LOOP ============================

while n_attacks < MAX_ATTACKS

    this_chunk = min(CHUNK_SIZE, MAX_ATTACKS - n_attacks);
    if this_chunk <= 0
        break;
    end

    fprintf('--- chunk: current=%d  adding=%d\n', n_attacks, this_chunk);

    % Per-chunk temporary storage
    psteps   = zeros(this_chunk, 1, 'uint32');
    hits_mat = zeros(this_chunk, nNodes, 'uint16');

    % --------- PARFOR ATTACK SIMULATION ---------
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

    % --------- ACCUMULATE RESULTS ---------
    n_attacks = n_attacks + this_chunk;
    counts_per_step = counts_per_step + uint64(accumarray(double(psteps), 1, [nNodes,1]));

    for a = 1:this_chunk
        ps = double(psteps(a));
        part_counts(ps,:) = part_counts(ps,:) + uint64(hits_mat(a,:));
    end

    % --------- COMPUTE PiP MATRIX ---------
    node_P = zeros(nNodes, nNodes, 'double');
    for p = 1:nNodes
        c = double(counts_per_step(p));
        if c > 0
            node_P(p,:) = double(part_counts(p,:)) / c;
        else
            node_P(p,:) = NaN;
        end
    end

    % --------- HW95 METRIC ---------
    hw_mat = wilson_hw_matrix(part_counts, counts_per_step);
    hw95   = prctile(hw_mat(:), 95);

    attacks_hist(end+1,1) = n_attacks;
    hw95_hist(end+1,1)    = hw95;

    % --------- PLATEAU CHECK ---------
    [is_plateau, slope_val, range_val, mean_val] = ...
        check_hw_plateau(attacks_hist, hw95_hist, ...
                         plateau_start, range_tol, slope_tol, win_len);

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

out_dir = fullfile(PIP_ROOT, 'results_converge');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

[~, base] = fileparts(SUBJECT_FILE);
out_mat = fullfile(out_dir, sprintf('%s_ConvHW.mat', base));

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

save(out_mat, 'node_P', 'counts_per_step', 'part_counts', 'meta', '-v7.3');

fprintf('\nSaved converged PiP → %s\n', out_mat);


%% ===================== HELPERS ==============================

function hw_mat = wilson_hw_matrix(part_counts, counts_per_step)
% Compute Wilson 95% CI half-widths for all (depth,node) entries)

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
    p(isnan(p)) = 0;   % interpret no observations as 0 participation

    z2  = z^2;
    num = z .* sqrt((p .* (1-p) ./ n) + (z2 ./ (4*n^2)));
    den = 1 + (z2 / n);
    hw_mat(d,:) = num ./ den;
end
end

function [is_plateau, slope_val, range_val, mean_val] = ...
    check_hw_plateau(attacks_hist, hw95_hist, plateau_start, range_tol, slope_tol, win_len)
% Decide whether hw95 has reached a plateau:
%   1) magnitude: mean(hw95_window) < plateau_start
%   2) range:     max(hw95_window) - min(hw95_window) < range_tol
%   3) slope:     |slope(hw95 vs attacks)| < slope_tol

    n = numel(hw95_hist);
    if n < win_len
        is_plateau = false;
        slope_val = NaN;
        range_val = NaN;
        mean_val  = NaN;
        return;
    end

    idx = (n-win_len+1):n;
    hw_win  = hw95_hist(idx);
    att_win = attacks_hist(idx);

    mean_val  = mean(hw_win);
    range_val = max(hw_win) - min(hw_win);

    % linear fit slope
    P = polyfit(att_win, hw_win, 1);
    slope_val = P(1);

    mag_ok    = mean_val  < plateau_start;
    range_ok  = range_val < range_tol;
    slope_ok  = abs(slope_val) < slope_tol;

    is_plateau = mag_ok && range_ok && slope_ok;
end
