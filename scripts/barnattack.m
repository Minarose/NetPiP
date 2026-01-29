clear; clc;

PIP_ROOT = getenv("PIP_ROOT");
if strlength(PIP_ROOT) == 0
    PIP_ROOT = "/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats";
end
res_dir  = fullfile(PIP_ROOT, "results_converge");

% Adjust pattern if your files are named differently:
files = dir(fullfile(res_dir, "*_ConvHW.mat"));
if isempty(files)
    error("No *_ConvHW.mat files found in %s", res_dir);
end

nSubs = numel(files);
n_attacks_final = zeros(nSubs,1);
labels = strings(nSubs,1);

for i = 1:nSubs
    fpath = fullfile(res_dir, files(i).name);
    S = load(fpath, "meta");
    meta = S.meta;

    % If you later add meta.n_attacks_final, this will use it.
    if isfield(meta, "n_attacks_final")
        n_attacks_final(i) = meta.n_attacks_final;
    else
        % Fallback: last entry of attacks_hist = final n_attacks
        n_attacks_final(i) = meta.attacks_hist(end);
    end

    [~, base, ~] = fileparts(files(i).name);
    labels(i) = string(base);
end

figure;
bar(n_attacks_final);
xlabel('Subject');
ylabel('Number of attacks at convergence');
title('PiP convergence: nattacks per subject');
xticks(1:nSubs);
xticklabels(labels);
xtickangle(45);   % tilt labels so they don’t overlap
grid on;
