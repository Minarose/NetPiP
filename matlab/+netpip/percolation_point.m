function k = percolation_point(lcc, slcc) %#ok<INUSL>
%PERCOLATION_POINT Removals at the percolation point of a targeted attack.
%   First global maximum of the SLCC over k = 0..N-1 removals, including the
%   intact graph; after k >= 1 removals a lone survivor counts as SLCC = 1.
%   Returns NaN if the SLCC never exceeds 1.
second = slcc(1:end-1);
second(2:end) = max(second(2:end), 1);
[m, i] = max(second);
if m > 1, k = i - 1; else, k = NaN; end
end
