function [is_plateau, slope_val, range_val, mean_val] = plateau_reached(attacks_hist, hw95_hist, varargin)
%PLATEAU_REACHED Has the Wilson-95 half-width plateaued?
%
%   [is_plateau, slope, range, mean_val] = netpip.plateau_reached(...
%       attacks_hist, hw95_hist, 'HW95Tol', 0.05, 'RangeTol', 0.005, ...
%       'SlopeTol', 1e-7, 'Window', 5, 'EnforceHW', false);

    p = inputParser;
    p.addParameter('HW95Tol', 0.05);
    p.addParameter('RangeTol', 0.005);
    p.addParameter('SlopeTol', 1e-7);
    p.addParameter('Window', 5);
    p.addParameter('EnforceHW', false);
    p.parse(varargin{:});
    o = p.Results;

    n = numel(hw95_hist);
    if n < o.Window
        is_plateau = false;
        slope_val  = NaN; range_val = NaN; mean_val = NaN;
        return;
    end
    hw  = hw95_hist(end-o.Window+1:end);
    atk = attacks_hist(end-o.Window+1:end);
    mean_val  = mean(hw);
    range_val = max(hw) - min(hw);
    s = polyfit(atk, hw, 1);
    slope_val = s(1);
    if o.EnforceHW
        is_plateau = (mean_val < o.HW95Tol) && (range_val < o.RangeTol) && (abs(slope_val) < o.SlopeTol);
    else
        is_plateau = (range_val < o.RangeTol) && (abs(slope_val) < o.SlopeTol);
    end
end
