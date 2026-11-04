function [counts, hits, events] = accumulate_orders(A, orders, counts, hits)
%ACCUMULATE_ORDERS PiP counts for given removal orders (one 1-based order per row).
%   [counts, hits, events] = ACCUMULATE_ORDERS(A, orders) returns
%   counts(k)   = number of orders whose percolation event is at step k
%   hits(k, i)  = number of those in which node i was removed by step k
%   events(r)   = event step of order r.
%   Pass existing counts/hits to add to them.
nbr = netpip.neighbor_lists(A);
n = size(A,1);
if nargin < 3, counts = zeros(n,1); hits = zeros(n,n); end
events = zeros(size(orders,1), 1);
for r = 1:size(orders,1)
    o = orders(r,:);
    k = netpip.percolation_event(o, nbr);
    events(r) = k;
    counts(k) = counts(k) + 1;
    hits(k, o(1:k)) = hits(k, o(1:k)) + 1;
end
end
