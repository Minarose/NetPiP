function [lcc, slcc] = component_trace(A, order)
%COMPONENT_TRACE LCC and SLCC sizes after k = 0..N removals along order.
%   order is a 1-based permutation; lcc(k+1) and slcc(k+1) are the sizes
%   after k removals (removed nodes absent).
nbr = netpip.neighbor_lists(A);
n = numel(order);
parent = 1:n; sz = ones(1,n); active = false(1,n); freq = zeros(1,n);
lcc = zeros(1, n+1); slcc = zeros(1, n+1);
largest = 0;
for s = n:-1:1
    node = order(s); active(node) = true;
    freq(1) = freq(1) + 1; largest = max(largest, 1);
    for other = nbr{node}
        if ~active(other), continue; end
        r = node; while parent(r) ~= r, r = parent(r); end
        t = other; while parent(t) ~= t, t = parent(t); end
        if r == t, continue; end
        if sz(r) < sz(t), tmp = r; r = t; t = tmp; end
        freq(sz(r)) = freq(sz(r)) - 1; freq(sz(t)) = freq(sz(t)) - 1;
        parent(t) = r; sz(r) = sz(r) + sz(t); freq(sz(r)) = freq(sz(r)) + 1;
        largest = max(largest, sz(r));
    end
    k = s - 1;   % removals so far
    lcc(k+1) = largest;
    if freq(largest) >= 2
        slcc(k+1) = largest;
    else
        v = largest - 1;
        while v > 0 && freq(v) == 0, v = v - 1; end
        slcc(k+1) = v;
    end
end
end
