% netpip MATLAB quickstart: PiP on a small synthetic graph.
% Run from the repository root after `addpath('matlab')`.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

% Build a 'barbell' graph: two K4 cliques joined by one bridge edge.
n = 8;
A = zeros(n);
for i = 1:4
    for j = i+1:4
        A(i, j) = 1; A(j, i) = 1;
    end
end
for i = 5:8
    for j = i+1:8
        A(i, j) = 1; A(j, i) = 1;
    end
end
A(4, 5) = 1; A(5, 4) = 1;

fprintf('Validating input adjacency (read-only)...\n');
report = netpip.validate_adjacency(A, 'MinGiantFraction', 0.75);
disp(report);

fprintf('Running PiP Monte Carlo (small budget for the demo)...\n');
res = netpip.run_pip(A, ...
    'MaxAttacks', 2000, ...
    'ChunkSize', 500, ...
    'Seed', 0, ...
    'Verbose', true);
fprintf('  attacks=%d  converged=%d  elapsed=%.2fs\n', ...
    res.n_attacks, res.converged, res.elapsed_sec);

fprintf('\nTilted-peak ranking:\n');
[order, peak_step, peak_amp] = netpip.tilted_peak_rank(res.node_P);
for r = 1:numel(order)
    fprintf('  rank %2d  node %2d  peak_step=%2d  amp=%.4f\n', ...
        r, order(r), peak_step(order(r)), peak_amp(order(r)));
end

fprintf('\nPiP percolation point and top-n hub set:\n');
pp = netpip.percolation_point(A, order);
fprintf('  perc_point = %d  hub nodes = %s\n', pp, mat2str(order(1:pp)));
