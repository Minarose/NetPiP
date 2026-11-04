function out = critical_cluster(profiles, kRange)
%CRITICAL_CLUSTER Ward clustering of node PiP profiles (nodes x steps).
%   k maximises the mean silhouette over kRange (default 2:10; singletons
%   score 0, as in scikit-learn). The critical cluster has the largest mean
%   summed profile. Requires the Statistics and Machine Learning Toolbox.
if nargin < 2, kRange = 2:10; end
X = double(profiles);
Z = linkage(X, 'ward');
D = squareform(pdist(X));
best = -Inf; scores = nan(size(kRange));
for t = 1:numel(kRange)
    lab = cluster(Z, 'maxclust', kRange(t));
    scores(t) = mean_silhouette(D, lab);
    if scores(t) > best, best = scores(t); k = kRange(t); labels = lab; end
end
summed = sum(X, 2);
u = unique(labels); m = arrayfun(@(c) mean(summed(labels == c)), u);
[~, j] = max(m);
out = struct('k', k, 'silhouette', best, 'silhouette_by_k', scores, 'labels', labels, ...
             'critical_nodes', find(labels == u(j)));
end

function s = mean_silhouette(D, lab)
n = numel(lab); v = zeros(n,1); u = unique(lab);
for i = 1:n
    own = lab == lab(i);
    if sum(own) <= 1, continue; end
    a = sum(D(i, own)) / (sum(own) - 1);
    b = Inf;
    for c = u.'
        if c ~= lab(i), b = min(b, mean(D(i, lab == c))); end
    end
    v(i) = (b - a) / max(a, b);
end
s = mean(v);
end
