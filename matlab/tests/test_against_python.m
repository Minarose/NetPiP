%TEST_AGAINST_PYTHON Check the MATLAB toolbox against the Python reference outputs.
%   Run from the repository's matlab/ folder:
%       cd matlab; addpath(pwd); run tests/test_against_python.m
%   The reference files in ../tests/reference are written by
%   tests/make_reference.py. MATLAB and NumPy draw different random numbers,
%   so both toolboxes are given the same fixed removal orders and the same
%   saved counts, and every derived quantity must agree.
%   Requires the Statistics and Machine Learning Toolbox (critical_cluster).
here = fileparts(mfilename('fullpath'));
ref = fullfile(here, '..', '..', 'tests', 'reference');
root = fullfile(here, '..', '..');
addpath(fullfile(here, '..'));
expected = jsondecode(fileread(fullfile(ref, 'expected.json')));
nfail = 0;

% 1. Event kernel and counts on fixed orders (karate club).
E = readmatrix(fullfile(root, 'examples', 'example_data', 'karate_edges.csv'));
A = zeros(34); A(sub2ind([34 34], E(:,1), E(:,2))) = 1; A = max(A, A.');
orders = readmatrix(fullfile(ref, 'karate_orders.csv'));
[counts, hits, events] = netpip.accumulate_orders(A, orders);
nfail = nfail + check('events', events, readmatrix(fullfile(ref, 'karate_events.csv')));
nfail = nfail + check('counts', counts, readmatrix(fullfile(ref, 'karate_counts.csv')));
nfail = nfail + check('hits', hits, readmatrix(fullfile(ref, 'karate_hits.csv')));
[lcc, slcc] = netpip.component_trace(A, orders(1,:));
nfail = nfail + check('LCC trace', lcc(:), expected.karate.trace_order1_lcc(:));
nfail = nfail + check('SLCC trace', slcc(:), expected.karate.trace_order1_slcc(:));

% 2. Surfaces, scores, attack outcome, clustering and RMS from the paper's language counts.
L = expected.language;
c = readmatrix(fullfile(ref, 'language_counts.csv'));
h = readmatrix(fullfile(ref, 'language_hits.csv'));
nfail = nfail + check('retained steps', netpip.retained_steps(c)', L.retained_steps(:));
lin = netpip.pip_scores(c, h, 'linear');
nfail = nfail + check('linear scores', lin, L.linear_scores(:), 1e-12);
nfail = nfail + check('exponential scores', netpip.pip_scores(c, h, 'exponential'), L.exponential_scores(:), 1e-12);
o = netpip.attack_orders(lin);
nfail = nfail + check('PiP order', o(:), L.pip_order(:));
E = readmatrix(fullfile(root, 'data', 'networks', 'language', 'edges.csv'));
LA = zeros(66); LA(sub2ind([66 66], E(:,1), E(:,2))) = 1; LA = max(LA, LA.');
att = netpip.evaluate_attacks(LA, o);
nfail = nfail + check('PiP percolation point', att.peak_removals, L.pip_attack.peak_removals);
nfail = nfail + check('LCC area', att.lcc_area, L.pip_attack.lcc_area, 1e-12);
[~, W] = netpip.weighted_surface(c, h, 'linear');
cl = netpip.critical_cluster(W.');
nfail = nfail + check('k', cl.k, L.k);
nfail = nfail + check('critical nodes', sort(cl.critical_nodes(:)), L.critical_nodes(:));
nfail = nfail + check('silhouette', cl.silhouette, L.silhouette, 1e-9);
files = dir(fullfile(ref, 'language_snapshot_*_counts.csv'));
snaps = struct('attacks', {}, 'counts', {}, 'hits', {});
for f = files.'
    a = sscanf(f.name, 'language_snapshot_%d_counts.csv');
    snaps(end+1) = struct('attacks', a, 'counts', readmatrix(fullfile(ref, f.name)), ...
        'hits', readmatrix(fullfile(ref, strrep(f.name, '_counts', '_hits')))); %#ok<SAGROW>
end
T = netpip.rms_convergence(snaps);
nfail = nfail + check('RMS', T.rms, [L.rms.rms]', 1e-12);
H = expected.hospital_exponential_absolute_weights;
nfail = nfail + check('absolute weights', netpip.step_weights(H.steps, H.n, 'exponential', 'absolute'), H.weights(:), 1e-14);

% 3. Sampling runs end to end.
r = netpip.run_pip(A, 2000, 'Seed', 1, 'Checkpoints', 1000);
nfail = nfail + check('sampled total', sum(r.counts), 2000);
nfail = nfail + check('hit conservation', sum(r.hits, 2), r.counts .* (1:34)');

if nfail == 0
    fprintf('\nAll MATLAB checks match the Python reference.\n');
else
    error('%d MATLAB checks failed.', nfail);
end

function bad = check(name, got, want, tol)
if nargin < 4, tol = 0; end
got = double(got(:)); want = double(want(:));
ok = isequal(size(got), size(want)) && all(abs(got - want) <= tol * max(1, abs(want)));
if ok, fprintf('  ok    %s\n', name); else, fprintf('  FAIL  %s\n', name); end
bad = ~ok;
end
