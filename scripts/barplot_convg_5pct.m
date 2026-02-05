% ================================================================
% plot_pip_plateau_summary_5pct.m
% ================================================================
% Summary plots for 5% thresholded convergence runs.
% ================================================================

clear; clc;

PIP_ROOT = getenv("PIP_ROOT");
if strlength(PIP_ROOT) == 0
    PIP_ROOT = "/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats";
end
res_dir  = fullfile(PIP_ROOT, "results_converge_5pct");

files = dir(fullfile(res_dir, "*_ConvHW.mat"));
if isempty(files)
    error("No *_ConvHW.mat files found in %s", res_dir);
end

nSubj = numel(files);

all_attacks   = cell(nSubj,1);
all_hw95      = cell(nSubj,1);
final_attacks = zeros(nSubj,1);
elapsed_min   = zeros(nSubj,1);
labels        = strings(nSubj,1);

for i = 1:nSubj
    fpath = fullfile(res_dir, files(i).name);
    S = load(fpath, "meta");
    meta = S.meta;

    attacks = meta.attacks_hist(:);
    hw95    = meta.hw95_hist(:);

    all_attacks{i} = attacks;
    all_hw95{i}    = hw95;
    final_attacks(i) = attacks(end);
    elapsed_min(i)   = meta.elapsed_sec / 60;

    [~, base] = fileparts(meta.subject_file);
    labels(i) = string(base);
end

out_dir = "/hpf/projects/dkadis/ismail/NetPiP/figures/5percthresh_analysis/convergence";
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

figure('Name','PiP convergence – HW95 curves','Color','w'); hold on;
for i = 1:nSubj
    plot(all_attacks{i}, all_hw95{i}, '-o', 'LineWidth', 1);
end
xlabel('Number of attacks');
ylabel('HW95 (95% CI half-width)');
title('PiP convergence (HW95) across subjects (5% threshold)');
legend(labels, 'Interpreter','none', 'Location','eastoutside');
grid on;
saveas(gcf, fullfile(out_dir, "PiP_HW95_convergence_all_subjects.png"));
saveas(gcf, fullfile(out_dir, "PiP_HW95_convergence_all_subjects.pdf"));

figure('Name','#Attacks at convergence','Color','w');
bar(final_attacks);
xticks(1:nSubj);
xticklabels(labels);
xtickangle(45);
ylabel('Number of attacks at convergence');
title('PiP: attacks at plateau (5% threshold)');
grid on;
saveas(gcf, fullfile(out_dir, "PiP_attacks_at_convergence_bar.png"));
saveas(gcf, fullfile(out_dir, "PiP_attacks_at_convergence_bar.pdf"));

figure('Name','Runtime per subject','Color','w');
bar(elapsed_min);
xticks(1:nSubj);
xticklabels(labels);
xtickangle(45);
ylabel('Runtime (minutes)');
title('PiP convergence runtime per subject (5% threshold)');
grid on;
saveas(gcf, fullfile(out_dir, "PiP_runtime_per_subject_bar.png"));
saveas(gcf, fullfile(out_dir, "PiP_runtime_per_subject_bar.pdf"));
