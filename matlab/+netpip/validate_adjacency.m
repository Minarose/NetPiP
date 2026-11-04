function A = validate_adjacency(A)
%VALIDATE_ADJACENCY Check that A is a valid PiP input and return it as double.
%   A must be square, binary (0/1), symmetric (undirected) with a zero
%   diagonal. Disconnected graphs and isolated nodes are allowed.
A = full(double(A));
if ~ismatrix(A) || size(A,1) ~= size(A,2)
    error('netpip:adjacency', 'Adjacency must be a square N x N matrix.');
end
if size(A,1) < 3
    error('netpip:adjacency', 'Adjacency needs at least 3 nodes.');
end
if any(A(:) ~= 0 & A(:) ~= 1)
    error('netpip:adjacency', 'Adjacency must be binary (0/1); threshold weighted matrices first.');
end
if ~isequal(A, A.')
    error('netpip:adjacency', 'Adjacency must be symmetric (undirected).');
end
if any(diag(A))
    error('netpip:adjacency', 'Adjacency must have a zero diagonal (no self-loops).');
end
end
