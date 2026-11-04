function orders = attack_orders(scores, nTieOrders, seed)
%ATTACK_ORDERS Removal orders by descending score with random tie-breaking.
%   Scores are rounded to 12 decimals. If ties remain, nTieOrders (default
%   500) orders are returned (one per row, 1-based), otherwise one.
%   MATLAB's tie-breaking draws differ from Python's; untied orders match.
if nargin < 2 || isempty(nTieOrders), nTieOrders = 500; end
if nargin < 3 || isempty(seed), seed = 860024; end
r = round(scores(:).', 12);
n = numel(r);
reps = 1; if numel(unique(r)) < n, reps = nTieOrders; end
rng(seed);
orders = zeros(reps, n);
for t = 1:reps
    sh = randperm(n);
    [~, idx] = sort(-r(sh), 'ascend');   % MATLAB sort is stable
    orders(t,:) = sh(idx);
end
end
