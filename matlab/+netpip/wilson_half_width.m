function hw_mat = wilson_half_width(part_counts, counts_per_step)
%WILSON_HALF_WIDTH Per-cell Wilson 95% half-width on binomial proportions.
%
%   hw = netpip.wilson_half_width(part_counts, counts_per_step) returns a
%   (nDepths x nNodes) matrix of Wilson 95% half-widths used for PiP
%   convergence diagnostics. Rows where counts_per_step(d) == 0 are NaN.

    z = 1.96;
    [nDepths, nNodes] = size(part_counts);
    hw_mat = zeros(nDepths, nNodes);
    for d = 1:nDepths
        n = double(counts_per_step(d));
        if n == 0
            hw_mat(d, :) = NaN;
            continue;
        end
        k  = double(part_counts(d, :));
        p  = k ./ n;
        p(isnan(p)) = 0;
        z2  = z^2;
        num = z .* sqrt((p .* (1-p) ./ n) + (z2 ./ (4*n^2)));
        den = 1 + (z2 / n);
        hw_mat(d, :) = num ./ den;
    end
end
