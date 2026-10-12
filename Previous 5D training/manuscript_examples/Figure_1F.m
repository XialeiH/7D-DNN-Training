function Figure_1F(outputDir)
% Plot only the timing panel from original Torch TSVs, without PDF images.
% Source: /scratch/xh2906/librarySCI_runs/figure1g2_dense_cold_equilibrium_20260829/results
% Error bars retain the source definition: sample SD / sqrt(number of trials).
if nargin < 1
    outputDir = fileparts(mfilename('fullpath'));
end
dataRoot = ['/Users/xialeihuang/Desktop/Neuroscience_Project/' ...
    'Spectral_Analysis/matlab-inserting_into_CG_model/report_figure_artifacts/' ...
    'Figure_1G/dense_cold_exact_results'];
files = [dir(fullfile(dataRoot, 'cg_dense_*HC_repeat_*.tsv')); ...
    dir(fullfile(dataRoot, 'dnn_fast_*HC_repeat_*.tsv'))];
snnFiles = dir(fullfile(dataRoot, 'snn_paper2_exact_*HC_repeat_*.tsv'));
assert(~isempty(files) && numel(snnFiles) == 5);
parts = cell(numel(files), 1);
for k = 1:numel(files)
    parts{k} = readtable(fullfile(files(k).folder, files(k).name), ...
        'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string');
end
T = vertcat(parts{:});
parts = cell(numel(snnFiles), 1);
for k = 1:numel(snnFiles)
    parts{k} = readtable(fullfile(snnFiles(k).folder, snnFiles(k).name), ...
        'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string');
end
S = vertcat(parts{:});
fieldSizes = [3 4 6 8 10 20 30 40];
expectedCG = [100 100 20 20 20 20 5 5];
expectedDNN = [100 100 20 20 20 20 20 20];
for k = 1:numel(fieldSizes)
    rows = T.FieldHC == fieldSizes(k);
    assert(sum(rows & T.Method == "CG") == expectedCG(k));
    assert(sum(rows & T.Method == "DNN surrogate") == expectedDNN(k));
end
assert(all(T.Converged));
assert(all(S.CPUs == 16) && all(S.Steps == 100000) && ...
    all(S.SimulatedMs == 10000) && all(S.FieldHC == 3));

colors = [178 24 43; 227 74 51; 254 227 145; ...
    171 217 233; 67 147 195; 33 102 172] / 255;
cgColor = colors(1, :); dnnColor = colors(6, :); snnColor = colors(3, :);
fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Renderer', 'painters');
textScale = 51/36;
fig.Position = [0.4 0.4 textScale*[11.2 9.1]];
% Expand the existing layout to keep the enlarged labels separated.
panelPosition = textScale*[1.7 1.5 8.525 6.35555555555556];
ax = axes(fig, 'Units', 'inches', 'Position', panelPosition, ...
    'PositionConstraint', 'innerposition');
hold(ax, 'on');
x = 1:numel(fieldSizes);
cgMean = zeros(size(x)); cgSem = zeros(size(x));
dnnMean = zeros(size(x)); dnnSem = zeros(size(x));
for k = 1:numel(fieldSizes)
    cg = T.Seconds(T.FieldHC == fieldSizes(k) & T.Method == "CG");
    dnn = T.Seconds(T.FieldHC == fieldSizes(k) & T.Method == "DNN surrogate");
    scatter(ax, x(k)*ones(size(cg)), cg, 42, cgColor, 'filled', ...
        'MarkerFaceAlpha', 0.42, 'MarkerEdgeAlpha', 0.42, 'HandleVisibility', 'off');
    scatter(ax, x(k)*ones(size(dnn)), dnn, 42, dnnColor, 'filled', ...
        'MarkerFaceAlpha', 0.42, 'MarkerEdgeAlpha', 0.42, 'HandleVisibility', 'off');
    cgMean(k) = mean(cg); cgSem(k) = std(cg)/sqrt(numel(cg));
    dnnMean(k) = mean(dnn); dnnSem(k) = std(dnn)/sqrt(numel(dnn));
end
snn = S.Seconds(S.Method == "SNN" & S.FieldHC == 3);
assert(numel(snn) == 5);
scatter(ax, ones(size(snn)), snn, 66, snnColor, 'd', 'filled', ...
    'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 1, ...
    'MarkerFaceAlpha', 0.62, 'MarkerEdgeAlpha', 0.72, 'HandleVisibility', 'off');
snnMean = mean(snn); snnSem = std(snn)/sqrt(numel(snn));
hCG = plot(ax, x, cgMean, '-o', 'Color', cgColor, ...
    'MarkerFaceColor', cgColor, 'MarkerSize', 8, 'LineWidth', 2.8);
hDNN = plot(ax, x, dnnMean, '-o', 'Color', dnnColor, ...
    'MarkerFaceColor', dnnColor, 'MarkerSize', 8, 'LineWidth', 2.8);
hSNN = plot(ax, 1, snnMean, 'd', 'Color', [0.2 0.2 0.2], ...
    'MarkerFaceColor', snnColor, 'MarkerSize', 10, 'LineWidth', 1.4);
errorbar(ax, x, cgMean, cgSem, 'LineStyle', 'none', 'Color', cgColor, ...
    'LineWidth', 2.2, 'CapSize', 12, 'HandleVisibility', 'off');
errorbar(ax, x, dnnMean, dnnSem, 'LineStyle', 'none', 'Color', dnnColor, ...
    'LineWidth', 2.2, 'CapSize', 12, 'HandleVisibility', 'off');
errorbar(ax, 1, snnMean, snnSem, 'LineStyle', 'none', 'Color', [0.2 0.2 0.2], ...
    'LineWidth', 2.2, 'CapSize', 12, 'HandleVisibility', 'off');

allTimes = [T.Seconds(ismember(T.FieldHC, fieldSizes)); snn];
lowestDecade = floor(log10(min(allTimes)));
decades = lowestDecade:floor(log10(max(allTimes)));
set(ax, 'YScale', 'log', ...
    'YLim', [10^lowestDecade 10^(log10(max(allTimes))+0.15)], ...
    'YTick', 10.^decades, ...
    'YTickLabel', arrayfun(@(d)sprintf('$10^{%d}$', d), decades, 'UniformOutput', false), ...
    'TickLabelInterpreter', 'latex', 'XLim', [0.55 numel(fieldSizes)+0.45], ...
    'XTick', x, 'XTickLabel', arrayfun(@(v)sprintf('$%d*%d$', v, v), ...
        fieldSizes, 'UniformOutput', false), ...
    'FontName', 'Arial', 'FontSize', 51, 'FontWeight', 'normal', 'LineWidth', 1.8, ...
    'TickDir', 'out', 'Box', 'off');
xtickangle(ax, 30);
xlabel(ax, 'Field Size', 'FontName', 'Arial', 'FontSize', 61, 'FontWeight', 'bold');
ylabel(ax, 'End-to-end equilibrium time (s)', 'FontName', 'Arial', 'FontSize', 61, 'FontWeight', 'bold');
grid(ax, 'on'); ax.GridAlpha = 0.14;
ax.Toolbar.Visible = 'off';
lgd = legend(ax, [hCG hDNN hSNN], {'CG', 'DNN surrogate', 'SNN'}, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontName', 'Arial', ...
    'FontSize', 51, 'FontWeight', 'bold', 'Location', 'northoutside');
lgd.ItemTokenSize = [32 18];
drawnow;
ax.Position = panelPosition;
lgd.Units = 'inches';
lp = lgd.Position;
lp(1) = panelPosition(1) + (panelPosition(3)-lp(3))/2;
lp(2) = panelPosition(2) + panelPosition(4) + 0.3;
lgd.Position = lp;
drawnow;
assert(max(abs(ax.Position(3:4)-panelPosition(3:4))) < 1e-8);
fprintf('CG/DNN/SNN points: %d/%d/%d\n', sum(expectedCG), sum(expectedDNN), numel(snn));
savefig(fig, fullfile(outputDir, 'Figure 1F.fig'));
exportgraphics(fig, fullfile(outputDir, 'Figure 1F.pdf'), ...
    'ContentType', 'vector', 'BackgroundColor', 'white', 'Padding', 16);
close(fig);
end
