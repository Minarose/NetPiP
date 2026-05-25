function pp = percolation_point(A, attack_order)
%PERCOLATION_POINT 1-based removal step at which the 2nd component first peaks.
%
%   pp = netpip.percolation_point(A, attack_order) mirrors the manuscript
%   convention: the 2nd-largest component size is recorded BEFORE each
%   removal. If no removal produces a second component larger than 1, returns
%   N (number of nodes).

    n = size(A, 1);
    if numel(attack_order) ~= n
        error('netpip:perc:length', 'attack_order length %d != n %d.', numel(attack_order), n);
    end
    A = double(A);
    second = zeros(1, n);
    for i = 1:n
        if nnz(A) == 0
            second(i) = 0;
        else
            G = graph(A);
            cc = conncomp(G);
            K = max(cc);
            if K < 2
                second(i) = 0;
            else
                h = histcounts(cc, 1:(double(K)+1));
                h = sort(h, 'descend');
                second(i) = h(2);
            end
        end
        node = attack_order(i);
        A(node, :) = 0;
        A(:, node) = 0;
    end
    if max(second) > 1
        pp = find(second == max(second), 1, 'first');
    else
        pp = n;
    end
end
