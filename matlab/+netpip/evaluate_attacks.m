function out = evaluate_attacks(A, orders)
%EVALUATE_ATTACKS Mean outcome of ordered attacks (one 1-based order per row).
%   out.peak_removals      mean removals to the percolation point
%   out.peak_removals_sd   s.d. across tie orders
%   out.lcc_area           normalized area under the LCC curve
%   out.removals_to_10pct  removals until the LCC is at most 10% of N
n = size(A,1);
m = size(orders,1);
peak = nan(m,1); area = zeros(m,1); r10 = zeros(m,1);
for t = 1:m
    [lcc, slcc] = netpip.component_trace(A, orders(t,:));
    peak(t) = netpip.percolation_point(lcc, slcc);
    area(t) = trapz((0:n)/n, lcc/n);
    r10(t) = find(lcc <= 0.1*n, 1) - 1;
end
out = struct('peak_removals', mean(peak, 'omitnan'), 'peak_removals_sd', std(peak, 1, 'omitnan'), ...
             'lcc_area', mean(area), 'removals_to_10pct', mean(r10), 'n_orders', m);
end
