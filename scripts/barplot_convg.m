% ================================================================
% plot_pip_plateau_summary_all_subjects.m
% ================================================================
% Loads all PiP convergence result files and:
%   1) Plots HW95 vs #attacks for each subject (line plot)
%   2) Plots a bar plot of final #attacks at convergence
%   3) Plots a bar plot of elapsed runtime (minutes)
%
% Assumes each .mat file contains a struct "meta" with fields:
%   meta.subject_file
%   meta.attacks_hist  (vector)
%   meta.hw95_hist     (vector)
%   meta.elapsed_sec   (scalar)
% ================================================================

clear; clc;

% ---------- Paths ----------
PIP_ROOT = getenv("PIP_ROOT");
if strlength(PIP_ROOT) == 0
    PIP_ROOT = "/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats";
end
res_dir  = fullfile(PIP_ROOT, "results_converge");

% Try plateau-style filenames first 
files = dir(fullfile(res_dir, "*_ConvHW.mat"));
if isempty(files)
    error("No *_ConvHW.mat files found in %s", res_dir);
end

if isempty(files)
    error("No PiP convergence result files found in %s", res_dir);
end

nSubj = numel(files);

all_attacks   = cell(nSubj,1);
all_hw95      = cell(nSubj,1);
final_attacks = zeros(nSubj,1);
elapsed_min   = zeros(nSubj,1);
labels        = strings(nSubj,1);

% ---------- Load all subjects ----------
for i = 1:nSubj
    fpath = fullfile(res_dir, files(i).name);
    S = load(fpath, "meta");
    meta = S.meta;

    attacks = meta.attacks_hist(:);
    hw95    = meta.hw95_hist(:);

    all_attacks{i} = attacks;
    all_hw95{i}    = hw95;

    final_attacks(i) = attacks(end);           % <-- #attacks when it stopped
    elapsed_min(i)   = meta.elapsed_sec / 60;  % <-- runtime in minutes

    [~, base] = fileparts(meta.subject_file);
    labels(i) = string(base);
end

%% ---------- 1) Convergence curves (HW95 vs attacks) ----------
figure('Name','PiP convergence – HW95 curves','Color','w'); hold on;

for i = 1:nSubj
    plot(all_attacks{i}, all_hw95{i}, '-o', 'LineWidth', 1);
end

xlabel('Number of attacks');
ylabel('HW95 (95% CI half-width)');
title('PiP convergence (HW95) across subjects');
legend(labels, 'Interpreter','none', 'Location','eastoutside');
grid on;

out_png = fullfile(res_dir, "PiP_HW95_convergence_all_subjects.png");
out_pdf = fullfile(res_dir, "PiP_HW95_convergence_all_subjects.pdf");
saveas(gcf, out_png);
saveas(gcf, out_pdf);

fprintf("Saved HW95 convergence plot:\n  %s\n  %s\n", out_png, out_pdf);

%% ---------- 2) Bar plot – final #attacks at stop ----------
figure('Name','#Attacks at convergence','Color','w');

bar(final_attacks);
xticks(1:nSubj);
xticklabels(labels);
xtickangle(45);
ylabel('Number of attacks at convergence');
title('PiP: attacks at plateau (per subject)');
grid on;

out_png2 = fullfile(res_dir, "PiP_attacks_at_convergence_bar.png");
out_pdf2 = fullfile(res_dir, "PiP_attacks_at_convergence_bar.pdf");
saveas(gcf, out_png2);
saveas(gcf, out_pdf2);

fprintf("Saved attacks-at-convergence bar plot:\n  %s\n  %s\n", out_png2, out_pdf2);

%% ---------- 3) Bar plot – runtime per subject (minutes) ----------
figure('Name','Runtime per subject','Color','w');

bar(elapsed_min);
xticks(1:nSubj);
xticklabels(labels);
xtickangle(45);
ylabel('Runtime (minutes)');
title('PiP convergence runtime per subject');
grid on;

out_png3 = fullfile(res_dir, "PiP_runtime_per_subject_bar.png");
out_pdf3 = fullfile(res_dir, "PiP_runtime_per_subject_bar.pdf");
saveas(gcf, out_png3);
saveas(gcf, out_pdf3);

fprintf("Saved runtime bar plot:\n  %s\n  %s\n", out_png3, out_pdf3);
