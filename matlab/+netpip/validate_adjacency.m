function report = validate_adjacency(A, varargin)
%VALIDATE_ADJACENCY Read-only check that A satisfies PiP's input contract.
%
%   report = netpip.validate_adjacency(A) verifies that A is a binary,
%   symmetric, zero-diagonal, sparse undirected adjacency matrix with a
%   sufficiently large giant connected component. The matrix A is NEVER
%   modified.
%
%   report = netpip.validate_adjacency(A, 'MinGiantFraction', 0.5) sets the
%   minimum allowed giant-component size as a fraction of N (default 0.5).
%
%   The returned `report` struct has fields:
%       n_nodes, n_edges, density, giant_component_size,
%       giant_component_fraction
%
%   If A does not satisfy the contract, this function calls error(...) with
%   an explanatory message and identifier 'netpip:validation:<reason>'.
%
%   See also NETPIP.RUN_PIP.

    p = inputParser;
    p.addParameter('MinGiantFraction', 0.5, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
    p.parse(varargin{:});
    minGiantFrac = p.Results.MinGiantFraction;

    if ~isnumeric(A) && ~islogical(A)
        error('netpip:validation:dtype', 'Adjacency must be numeric or logical.');
    end
    A = double(A);
    if ndims(A) ~= 2
        error('netpip:validation:shape', 'Adjacency must be 2D; got %dD.', ndims(A));
    end
    [n1, n2] = size(A);
    if n1 ~= n2
        error('netpip:validation:shape', 'Adjacency must be square; got [%d x %d].', n1, n2);
    end
    if n1 < 2
        error('netpip:validation:shape', 'Adjacency must have at least 2 nodes; got n=%d.', n1);
    end
    n = n1;

    if any(~isfinite(A(:)))
        error('netpip:validation:nonfinite', ...
            'Adjacency contains non-finite values (NaN/inf).');
    end

    u = unique(A(:));
    isBinary = all(ismember(u, [0; 1]));
    if ~isBinary
        error('netpip:validation:binary', ...
            'Adjacency must be binary {0, 1}. Binarize first, e.g. A = double(W > 0).');
    end

    if ~isequal(A, A.')
        error('netpip:validation:symmetric', ...
            'Adjacency must be symmetric (A == A.''). Symmetrize first.');
    end

    if any(diag(A) ~= 0)
        error('netpip:validation:diagonal', ...
            'Adjacency must have zero diagonal. Use A(1:n+1:end) = 0.');
    end

    n_edges = nnz(triu(A, 1));
    maxEdges = n * (n - 1) / 2;
    density = n_edges / maxEdges;
    if density >= 1
        error('netpip:validation:complete', ...
            'Adjacency is the complete graph (density == 1).');
    end
    if n_edges == 0
        error('netpip:validation:empty', ...
            'Adjacency has no edges (density == 0).');
    end

    G = graph(A);
    [bins, sizes] = conncomp(G, 'OutputForm', 'cell'); %#ok<ASGLU>
    csizes = cellfun(@numel, sizes);
    gcc = max(csizes);
    gccFrac = gcc / n;
    if gccFrac < minGiantFrac
        error('netpip:validation:giant', ...
            'Largest connected component has %d/%d nodes (%.1f%% < %.1f%% required).', ...
            gcc, n, 100*gccFrac, 100*minGiantFrac);
    end

    report = struct( ...
        'n_nodes', n, ...
        'n_edges', n_edges, ...
        'density', density, ...
        'giant_component_size', gcc, ...
        'giant_component_fraction', gccFrac);
end
