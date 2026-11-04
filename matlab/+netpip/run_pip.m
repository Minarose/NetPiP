function result = run_pip(A, nAttacks, varargin)
%RUN_PIP Participation in Percolation by Monte Carlo sampling of random attacks.
%   result = netpip.run_pip(A, nAttacks) samples nAttacks uniformly random
%   node-removal orders of the binary, undirected adjacency matrix A and
%   records, for each attack, which nodes had been removed when the network
%   reached its percolation point (first maximum of the second-largest
%   connected component).
%
%   Name-value options:
%     'Seed'        - seed for rng (default: current generator state)
%     'Checkpoints' - attack totals at which to store a copy of counts/hits
%     'BatchSize'   - attacks per batch (default 10000)
%     'UseParfor'   - true to process each batch with parfor (default false)
%     'Verbose'     - print progress (default false)
%
%   result fields: counts (N x 1), hits (N x N), nAttacks, seed, snapshots
%   (struct array with fields attacks, counts, hits).
%
%   Note: MATLAB's random number generator differs from NumPy's, so MATLAB
%   runs are statistically equivalent to, but not bit-identical with, the
%   Python runs used in the paper. Use the Python toolbox to reproduce the
%   paper's counts exactly.
p = inputParser;
addParameter(p, 'Seed', []);
addParameter(p, 'Checkpoints', []);
addParameter(p, 'BatchSize', 10000);
addParameter(p, 'UseParfor', false);
addParameter(p, 'Verbose', false);
parse(p, varargin{:});
opt = p.Results;
A = netpip.validate_adjacency(A);
n = size(A,1);
nbr = netpip.neighbor_lists(A);
if ~isempty(opt.Seed), rng(opt.Seed); end
marks = sort(unique(opt.Checkpoints(opt.Checkpoints > 0 & opt.Checkpoints <= nAttacks)));
counts = zeros(n,1); hits = zeros(n,n);
snapshots = struct('attacks', {}, 'counts', {}, 'hits', {});
done = 0;
while done < nAttacks
    nextMark = marks(find(marks > done, 1));
    if isempty(nextMark), nextMark = nAttacks; end
    b = min(opt.BatchSize, nextMark - done);
    orders = zeros(b, n);
    for r = 1:b, orders(r,:) = randperm(n); end
    events = zeros(b,1);
    if opt.UseParfor
        parfor r = 1:b
            events(r) = netpip.percolation_event(orders(r,:), nbr); %#ok<PFBNS>
        end
    else
        for r = 1:b
            events(r) = netpip.percolation_event(orders(r,:), nbr);
        end
    end
    for r = 1:b
        k = events(r);
        counts(k) = counts(k) + 1;
        hits(k, orders(r,1:k)) = hits(k, orders(r,1:k)) + 1;
    end
    done = done + b;
    if any(marks == done)
        snapshots(end+1) = struct('attacks', done, 'counts', counts, 'hits', hits); %#ok<AGROW>
    end
    if opt.Verbose, fprintf('  %d / %d attacks\n', done, nAttacks); end
end
result = struct('counts', counts, 'hits', hits, 'nAttacks', nAttacks, 'seed', opt.Seed, ...
                'snapshots', snapshots);
end
