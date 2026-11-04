function steps = retained_steps(counts, rule)
%RETAINED_STEPS 1-based removal steps kept for analysis.
%   'longest_block' (default): longest run of consecutive observed steps.
%   'all_observed': every step at which at least one attack percolated.
if nargin < 2, rule = 'longest_block'; end
obs = find(counts(:) > 0).';
if isempty(obs), error('netpip:counts', 'No attacks recorded.'); end
if strcmp(rule, 'all_observed'), steps = obs; return; end
breaks = [0, find(diff(obs) > 1), numel(obs)];
[~, j] = max(diff(breaks));
steps = obs(breaks(j)+1:breaks(j+1));
end
