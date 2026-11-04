function [steps, W] = weighted_surface(counts, hits, kind, rule, basis)
%WEIGHTED_SURFACE Weighted PiP surface W(j, i) = w_j * hits(k_j, i) / counts(k_j).
%   Rows are the retained steps (returned in steps, 1-based), columns nodes.
if nargin < 3 || isempty(kind), kind = 'linear'; end
if nargin < 4 || isempty(rule), rule = 'longest_block'; end
if nargin < 5 || isempty(basis), basis = 'retained'; end
steps = netpip.retained_steps(counts, rule);
c = double(counts(:));
P = double(hits(steps, :)) ./ c(steps);   % each row divided by its attack count
w = netpip.step_weights(steps, numel(counts), kind, basis);
W = P .* w;
end
