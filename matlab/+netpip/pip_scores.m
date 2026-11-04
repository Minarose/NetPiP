function s = pip_scores(counts, hits, kind, rule, basis)
%PIP_SCORES Node PiP score: mean weighted participation across retained steps.
%   Higher scores mean the node is more often removed before the network
%   first splits. Rank nodes with netpip.attack_orders(s).
if nargin < 3 || isempty(kind), kind = 'linear'; end
if nargin < 4 || isempty(rule), rule = 'longest_block'; end
if nargin < 5 || isempty(basis), basis = 'retained'; end
[~, W] = netpip.weighted_surface(counts, hits, kind, rule, basis);
s = mean(W, 1).';
end
