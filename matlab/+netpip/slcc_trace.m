function second = slcc_trace(order, nbr)
%SLCC_TRACE SLCC size among surviving nodes after r = 1..N-1 removals.
%   Nodes are added back in reverse removal order with a union-find
%   structure (union by size), which reconstructs each surviving graph
%   exactly. Removed nodes are absent; the SLCC is 0 with one component.
n = numel(order);
parent = 1:n; sz = ones(1,n); active = false(1,n);
freq = zeros(1, n);          % freq(s) = number of components of size s
second = zeros(1, n-1);
largest = 0; comps = 0;
for s = n:-1:2
    node = order(s);
    active(node) = true;
    freq(1) = freq(1) + 1; comps = comps + 1; largest = max(largest, 1);
    for other = nbr{node}
        if ~active(other), continue; end
        r = node; while parent(r) ~= r, r = parent(r); end
        t = other; while parent(t) ~= t, t = parent(t); end
        if r == t, continue; end
        if sz(r) < sz(t), tmp = r; r = t; t = tmp; end
        freq(sz(r)) = freq(sz(r)) - 1;
        freq(sz(t)) = freq(sz(t)) - 1;
        parent(t) = r;
        sz(r) = sz(r) + sz(t);
        freq(sz(r)) = freq(sz(r)) + 1;
        comps = comps - 1;
        largest = max(largest, sz(r));
    end
    removals = s - 1;    % state = graph after removing order(1:s-1)
    if comps > 1
        if freq(largest) >= 2
            second(removals) = largest;
        else
            v = largest - 1;
            while v > 0 && freq(v) == 0, v = v - 1; end
            second(removals) = v;
        end
    end
end
end
