function nbr = neighbor_lists(A)
%NEIGHBOR_LISTS Cell array of neighbour indices (row vectors) for each node.
A = netpip.validate_adjacency(A);
n = size(A,1);
nbr = cell(1, n);
for i = 1:n
    nbr{i} = find(A(i,:));
end
end
