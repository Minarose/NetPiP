% render_brainnet_overlap_tifs.m
% -------------------------------------------------------------------------
% Render BrainNet Viewer .tif images for the PiP-cluster vs metric-prefix
% overlap .node files in results/avg_metric_overlap/.
%
% Requires: BrainNet Viewer on the MATLAB path. Either set
%   BRAINNET_PATH    : folder that contains BrainNet_MapCfg.m
% (e.g. ~/Downloads/BrainNetViewer_20191031), or add it to addpath manually
% before calling this script.
%
% The default surface mesh is BrainMesh_ICBM152.nv inside
% BRAINNET_PATH/Data/SurfTemplate. Override via env BRAINNET_SURFACE_NV.
% The display config is the existing brainnet_overlap_pip.mat.
% -------------------------------------------------------------------------

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
outDir   = fullfile(repoRoot, 'results', 'avg_metric_overlap');

bnPath = getenv('BRAINNET_PATH');
if isempty(bnPath)
    candidates = { ...
        fullfile(getenv('HOME'),'Downloads','BrainNetViewer_20191031'); ...
        fullfile(getenv('HOME'),'Downloads','BrainNet'); ...
        fullfile(getenv('HOME'),'BrainNetViewer'); ...
        '/Applications/BrainNetViewer'};
    for i = 1:numel(candidates)
        if isfolder(candidates{i})
            bnPath = candidates{i}; break;
        end
    end
end
if isempty(bnPath) || ~isfolder(bnPath)
    error(['BrainNet Viewer folder not found.\n' ...
           'Set env BRAINNET_PATH to the folder containing BrainNet_MapCfg.m']);
end
addpath(bnPath);

surfFile = getenv('BRAINNET_SURFACE_NV');
if isempty(surfFile)
    surfFile = fullfile(bnPath, 'Data', 'SurfTemplate', 'BrainMesh_ICBM152.nv');
end
if ~isfile(surfFile)
    error(['Surface mesh not found at:\n  %s\n' ...
           'Set env BRAINNET_SURFACE_NV to a valid .nv file.'], surfFile);
end

cfgFile = fullfile(outDir, 'brainnet_overlap_pip.mat');
if ~isfile(cfgFile)
    error('Missing display config:\n  %s', cfgFile);
end

cases = { ...
    'brainnet_overlap_pip_vs_degree.node',      'brainnet_overlap_pip_vs_degree.tif'      ; ...
    'brainnet_overlap_pip_vs_betweenness.node', 'brainnet_overlap_pip_vs_betweenness.tif' ; ...
    'brainnet_overlap_pip_vs_pagerank.node',    'brainnet_overlap_pip_vs_pagerank.tif'    ; ...
};

for k = 1:size(cases,1)
    nodeFile = fullfile(outDir, cases{k,1});
    tifFile  = fullfile(outDir, cases{k,2});
    if ~isfile(nodeFile)
        warning('Missing .node, skipping: %s', nodeFile); continue
    end
    fprintf('Rendering %s -> %s\n', nodeFile, tifFile);
    BrainNet_MapCfg(surfFile, nodeFile, cfgFile, tifFile);
    close all;
end

fprintf('Done. .tif files in %s\n', outDir);
