%QUICKSTART PiP on Zachary's karate club network (MATLAB).
%   Run from the repository's matlab/ folder:  addpath(pwd); run examples/quickstart.m
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
E = readmatrix(fullfile(here, '..', '..', 'examples', 'example_data', 'karate_edges.csv'));
A = zeros(34); A(sub2ind([34 34], E(:,1), E(:,2))) = 1; A = max(A, A.');

% Sample random attacks (the paper used 1e9 in Python; 1e5 is quick here).
result = netpip.run_pip(A, 1e5, 'Seed', 1, 'Checkpoints', [1e4 5e4]);

% Score and rank nodes by linear-weighted PiP.
scores = netpip.pip_scores(result.counts, result.hits, 'linear');
[~, rankOrder] = sort(scores, 'descend');
fprintf('Most critical nodes: %s\n', mat2str(rankOrder(1:5).'));

% Convergence: RMS change between checkpoints (paper threshold 0.005).
snaps = [result.snapshots, struct('attacks', result.nAttacks, 'counts', result.counts, 'hits', result.hits)];
disp(netpip.rms_convergence(snaps));

% Targeted attack ordered by PiP.
out = netpip.evaluate_attacks(A, netpip.attack_orders(scores));
fprintf('PiP-ordered attack reaches the percolation point after %g removals\n', out.peak_removals);
