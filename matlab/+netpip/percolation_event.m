function k = percolation_event(order, nbr)
%PERCOLATION_EVENT Removal step at which one random attack percolates.
%   k = PERCOLATION_EVENT(order, nbr) returns the first number of removals
%   k in 1..N-1 at which the second-largest connected component (SLCC) of
%   the surviving graph reaches its maximum size, or N if the SLCC never
%   exceeds one node. order is a 1-based removal order; nbr comes from
%   netpip.neighbor_lists. Identical definition to the Python kernel.
second = netpip.slcc_trace(order, nbr);
n = numel(order);
best = 1; k = n;
for r = 1:n-1
    if second(r) > best
        best = second(r);
        k = r;
    end
end
end
