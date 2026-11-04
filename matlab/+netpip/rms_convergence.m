function T = rms_convergence(snapshots, rule, basis)
%RMS_CONVERGENCE RMS change of the linear PiP surface between checkpoints.
%   snapshots: struct array with fields attacks, counts, hits (as returned
%   by run_pip, plus the final result appended). The final checkpoint fixes
%   the retained steps and weights; only steps observed at every checkpoint
%   are compared:  RMS_t = sqrt(mean((W_t - W_{t-1}).^2)).
%   Returns a table with attacks, previous_attacks, rms, compared_rows.
if nargin < 2 || isempty(rule), rule = 'longest_block'; end
if nargin < 3 || isempty(basis), basis = 'retained'; end
[~, idx] = sort([snapshots.attacks]);
snapshots = snapshots(idx);
final = snapshots(end);
n = numel(final.counts);
steps = netpip.retained_steps(final.counts, rule);
w = netpip.step_weights(steps, n, 'linear', basis);
common = true(numel(steps), 1);
for t = 1:numel(snapshots)
    common = common & (snapshots(t).counts(steps) > 0);
end
rows = steps(common);
W = cell(1, numel(snapshots));
for t = 1:numel(snapshots)
    c = double(snapshots(t).counts(:)); h = double(snapshots(t).hits);
    W{t} = h(rows, :) ./ c(rows) .* w(common);
end
m = numel(snapshots) - 1;
attacks = zeros(m,1); previous = zeros(m,1); rms = zeros(m,1);
for t = 2:numel(snapshots)
    d = W{t} - W{t-1};
    attacks(t-1) = snapshots(t).attacks; previous(t-1) = snapshots(t-1).attacks;
    rms(t-1) = sqrt(mean(d(:).^2));
end
T = table(attacks, previous, rms, repmat(sum(common), m, 1), ...
    'VariableNames', {'attacks', 'previous_attacks', 'rms', 'compared_rows'});
end
