function res = run_pip(A, varargin)
%RUN_PIP Monte Carlo Participation-in-Percolation (PiP) engine.
%
%   res = netpip.run_pip(A) runs PiP on a binary undirected adjacency matrix
%   A (N x N) and returns a struct with fields:
%       node_P           - (N x N) double, node_P(p, i) = P(node i removed
%                          by step p | percolation step == p)
%       counts_per_step  - (N x 1) uint64
%       part_counts      - (N x N) uint64
%       n_attacks        - total Monte Carlo attacks performed
%       attacks_hist     - cumulative attacks at each chunk boundary
%       hw95_hist        - 95th percentile Wilson half-width per chunk
%       converged        - true if plateau early-stop triggered
%       elapsed_sec      - wall-clock runtime
%       meta             - struct of run parameters
%
%   This is a clean (non-Slurm) wrapper around the engine used by
%   scripts/2_pip_converge.m. The matrix A is NEVER modified;
%   call netpip.validate_adjacency(A) first if you want explicit input
%   checking.
%
%   Name-Value parameters (defaults match the manuscript):
%       'MaxAttacks'      (1e6)   hard cap on attacks
%       'ChunkSize'       (1e4)   attacks between convergence checks
%       'Seed'            ([])    RNG seed (uses 'shuffle' if empty)
%       'EnforceHW95'     (false) require mean(HW) < HW95Tol for plateau
%       'HW95Tol'         (0.05)
%       'RangeTol'        (0.005)
%       'SlopeTol'        (1e-7)
%       'PlateauWindow'   (5)
%       'RequireStable'   (3)
%       'UseParfor'       (false) use parfor across attacks within a chunk
%       'Verbose'         (true)  print per-chunk progress
%
%   See also NETPIP.VALIDATE_ADJACENCY, NETPIP.TILTED_PEAK_RANK,
%   NETPIP.PERCOLATION_POINT.

    p = inputParser;
    p.addParameter('MaxAttacks', 1e6, @(x) isnumeric(x) && isscalar(x) && x > 0);
    p.addParameter('ChunkSize', 1e4, @(x) isnumeric(x) && isscalar(x) && x > 0);
    p.addParameter('Seed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    p.addParameter('EnforceHW95', false, @islogical);
    p.addParameter('HW95Tol', 0.05, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('RangeTol', 0.005, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('SlopeTol', 1e-7, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('PlateauWindow', 5, @(x) isnumeric(x) && isscalar(x) && x >= 3);
    p.addParameter('RequireStable', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    p.addParameter('UseParfor', false, @islogical);
    p.addParameter('Verbose', true, @islogical);
    p.parse(varargin{:});
    opts = p.Results;

    A = double(A);
    n = size(A, 1);

    if isempty(opts.Seed)
        rng('shuffle');
    else
        rng(opts.Seed, 'Threefry');
    end

    counts_per_step = zeros(n, 1, 'uint64');
    part_counts     = zeros(n, n, 'uint64');
    attacks_hist    = zeros(0, 1);
    hw95_hist       = zeros(0, 1);
    stableCount     = 0;
    n_attacks       = 0;
    converged       = false;

    t0 = tic;
    while n_attacks < opts.MaxAttacks
        this_chunk = min(opts.ChunkSize, opts.MaxAttacks - n_attacks);
        if this_chunk <= 0
            break;
        end

        psteps   = zeros(this_chunk, 1, 'uint32');
        hits_mat = zeros(this_chunk, n, 'uint16');

        if opts.UseParfor
            parfor a = 1:this_chunk
                [psteps(a), hits_mat(a,:)] = local_one_attack(A, n);
            end
        else
            for a = 1:this_chunk
                [psteps(a), hits_mat(a,:)] = local_one_attack(A, n);
            end
        end

        n_attacks = n_attacks + this_chunk;
        counts_per_step = counts_per_step + ...
            uint64(accumarray(double(psteps), 1, [n, 1]));

        for a = 1:this_chunk
            ps = double(psteps(a));
            part_counts(ps,:) = part_counts(ps,:) + uint64(hits_mat(a,:));
        end

        hw_mat = netpip.wilson_half_width(part_counts, counts_per_step);
        hw95   = prctile(hw_mat(:), 95);
        attacks_hist(end+1, 1) = n_attacks; %#ok<AGROW>
        hw95_hist(end+1, 1)    = hw95;      %#ok<AGROW>

        [is_plateau, slope_val, range_val, mean_val] = netpip.plateau_reached( ...
            attacks_hist, hw95_hist, ...
            'HW95Tol', opts.HW95Tol, 'RangeTol', opts.RangeTol, ...
            'SlopeTol', opts.SlopeTol, 'Window', opts.PlateauWindow, ...
            'EnforceHW', opts.EnforceHW95);

        if is_plateau
            stableCount = stableCount + 1;
        else
            stableCount = 0;
        end

        if opts.Verbose
            fprintf(['  chunk @ n_attacks=%d  hw95=%.4f  ' ...
                'mean=%.4f  range=%.5f  slope=%.2e  plateau=%d  stable=%d\n'], ...
                n_attacks, hw95, mean_val, range_val, slope_val, is_plateau, stableCount);
        end

        if stableCount >= opts.RequireStable
            converged = true;
            break;
        end
    end
    elapsed_sec = toc(t0);

    node_P = zeros(n, n, 'double');
    for q = 1:n
        c = double(counts_per_step(q));
        if c > 0
            node_P(q,:) = double(part_counts(q,:)) / c;
        else
            node_P(q,:) = NaN;
        end
    end

    res = struct( ...
        'node_P',          node_P, ...
        'counts_per_step', counts_per_step, ...
        'part_counts',     part_counts, ...
        'n_attacks',       n_attacks, ...
        'attacks_hist',    attacks_hist, ...
        'hw95_hist',       hw95_hist, ...
        'converged',       converged, ...
        'elapsed_sec',     elapsed_sec, ...
        'meta',            opts);
end

function [pstep, hits_row] = local_one_attack(A, n)
    atk  = randperm(n);
    mask = true(1, n);
    second_comp = zeros(1, n-1, 'uint32');
    for step = 1:n-1
        mask(atk(step)) = false;
        sub = A(mask, mask);
        if nnz(sub) == 0
            bin2 = 0;
        else
            G = graph(sub);
            cc = conncomp(G);
            K = max(cc);
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
        pstep = uint32(find(second_comp == smax, 1, 'first'));
    else
        pstep = uint32(n);
    end
    hits_row = zeros(1, n, 'uint16');
    hits_row(atk(1:pstep)) = 1;
end
