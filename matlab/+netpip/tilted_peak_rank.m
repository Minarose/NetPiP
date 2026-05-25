function [order, peak_step, peak_amp] = tilted_peak_rank(node_P, varargin)
%TILTED_PEAK_RANK Rank nodes by their time-tilted PiP peak amplitude.
%
%   [order, peak_step, peak_amp] = netpip.tilted_peak_rank(node_P) returns a
%   1-based node order (length N) where order(1) is the highest-ranked PiP
%   node. The default tilt is tau = max(2, S/6) with negative values clipped
%   to 0, matching the canonical manuscript configuration.
%
%   Name-Value:
%       'TauFactor'    (1/6)
%       'ClipNegative' (true)

    p = inputParser;
    p.addParameter('TauFactor', 1/6);
    p.addParameter('ClipNegative', true, @islogical);
    p.parse(varargin{:});
    o = p.Results;

    P = node_P;
    P = local_crop(P);
    if isempty(P)
        error('netpip:ranking:empty', 'node_P is empty after cropping.');
    end
    P(~isfinite(P)) = 0;
    if o.ClipNegative
        P(P < 0) = 0;
    end

    S = size(P, 1);
    tau = max(2, S * o.TauFactor);
    s = (1:S).';
    w = exp(-s ./ tau);
    Pt = P .* w;

    [peak_amp, peak_step] = max(Pt, [], 1);
    peak_amp = peak_amp(:);
    peak_step = peak_step(:);
    [~, order] = sortrows([-peak_amp, peak_step], [1 2]);
    order = order(:)';
end

function Pc = local_crop(P)
    if isempty(P)
        Pc = P; return;
    end
    row_ok = any(~isnan(P), 2);
    if ~any(row_ok)
        Pc = []; return;
    end
    d = diff([false; row_ok; false]);
    starts = find(d == 1);
    ends   = find(d == -1) - 1;
    lens   = ends - starts + 1;
    [~, imax] = max(lens);
    Pc = P(starts(imax):ends(imax), :);
end
