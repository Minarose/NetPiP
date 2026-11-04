function w = step_weights(steps, n, kind, basis)
%STEP_WEIGHTS Decreasing weights that emphasise early fragmentation.
%   basis 'retained' (default): j = 1..S over the S retained steps;
%     linear (S-j+1)/S, exponential exp(-j/max(2,S/6)).
%   basis 'absolute': the actual step k and N replace j and S
%     (used for the paper's hospital network).
%   kind 'linear' (default), 'exponential' or 'none'.
if nargin < 3 || isempty(kind), kind = 'linear'; end
if nargin < 4 || isempty(basis), basis = 'retained'; end
steps = steps(:);
switch basis
    case 'retained', j = (1:numel(steps)).'; S = numel(steps);
    case 'absolute', j = double(steps); S = double(n);
    otherwise, error('netpip:weights', 'basis must be retained or absolute');
end
switch kind
    case 'linear', w = (S - j + 1) / S;
    case 'exponential', w = exp(-j / max(2, S/6));
    case 'none', w = ones(size(j));
    otherwise, error('netpip:weights', 'kind must be linear, exponential or none');
end
end
