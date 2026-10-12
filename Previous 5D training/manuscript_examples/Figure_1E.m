function Figure_1E(root, nearZeroFile)
outputDir = fileparts(mfilename('fullpath'));
outputPdf = fullfile(outputDir, 'Figure 1E.pdf');
outputFig = fullfile(outputDir, 'Figure 1E.fig');
% Recreate Figure 1F with the first eigenvector replacing the eigencluster.

if nargin < 1
    root = '/Users/xialeihuang/Desktop/Neuroscience_Project/Spectral_Analysis/matlab-inserting_into_CG_model/report_figure_artifacts/1_Neural_Network_Surrogate/Figure_1F_1';
end

dnn = load(fullfile(root, 'dnn_angle0_contrast100.mat'), 'conditionData');
cg = load(fullfile(root, 'cg_3dlibrary_angle0_contrast100.mat'), ...
    'fixedPointMaps', 'eigValues', 'topSingularModeE');
dnnModes = load(fullfile(root, 'dnn_cluster10.mat'), 'topEigenmodeE');
cgModes = load(fullfile(root, 'cg_cluster_modes10.mat'), 'topEigenmodeE');
dnnSingular = load(fullfile(root, 'dnn_top_singular_mode.mat'), ...
    'dnnTopSingularModeE');
if nargin < 2
    nearZeroFile = ['/Users/xialeihuang/Desktop/Neuroscience_Project/' ...
    'Spectral_Analysis/matlab-inserting_into_CG_model/' ...
    'l6_next_stage_analysis_20260717/near_zero_clustering_20260722/' ...
        'results_full16/near_zero_eigenvalues_angle0p00_contrast100.mat'];
end
spectra = load(nearZeroFile, 'eigenvalueData');

fixedMaps = {dnn.conditionData.FixedPointMaps.E; cg.fixedPointMaps.E};
eigenvalues = {spectra.eigenvalueData.J_baseline; cg.eigValues};
firstEigenvectors = {dnnModes.topEigenmodeE(:, :, 1); ...
    cgModes.topEigenmodeE(:, :, 1)};
singularModes = {dnnSingular.dnnTopSingularModeE; cg.topSingularModeE};

fixedLimits = [min([fixedMaps{1}(:); fixedMaps{2}(:)]), ...
    max([fixedMaps{1}(:); fixedMaps{2}(:)])];
eigenLimit = max(abs([firstEigenvectors{1}(:); firstEigenvectors{2}(:)]));
eigenLimits = [-eigenLimit, eigenLimit];
singularLimit = max(abs([singularModes{1}(:); singularModes{2}(:)]));
singularLimits = [-singularLimit, singularLimit];

fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
    'Renderer', 'painters', 'DefaultAxesFontSize', 35, ...
    'DefaultTextFontSize', 45, 'DefaultAxesFontName', 'Arial', ...
    'DefaultTextFontName', 'Arial');
% Match the 7.6-inch plot width of Draft Figure 1B.
fig.Position = [0.25 0.25 45.5 20.7];
lefts = [1.2 12.4 22.6 33.8];
widths = repmat(7.6, 1, 4);
bottoms = [10.9 2.3];
height = 7.6;

fixedAxes = gobjects(2, 1);
distributionAxes = gobjects(2, 1);
eigenAxes = gobjects(2, 1);
singularAxes = gobjects(2, 1);
for row = 1:2
    fixedAxes(row) = local_heatmap(fig, [lefts(1) bottoms(row) widths(1) height], ...
        fixedMaps{row}, fixedLimits, row == 2);
    distributionAxes(row) = local_realpart_distribution(fig, [lefts(2) bottoms(row) widths(2) height], ...
        eigenvalues{row}, row == 2);
    eigenAxes(row) = local_heatmap(fig, [lefts(3) bottoms(row) widths(3) height], ...
        firstEigenvectors{row}, eigenLimits, row == 2);
    singularAxes(row) = local_heatmap(fig, [lefts(4) bottoms(row) widths(4) height], ...
        singularModes{row}, singularLimits, row == 2);
    singularAxes(row).YAxis.TickLabelGapOffset = -3;
end

fixedColorbar = local_colorbar(fixedAxes(1), fixedLimits, 'sp/s');
eigenColorbar = local_colorbar(eigenAxes(1), eigenLimits, 'sp/s');
singularColorbar = local_colorbar(singularAxes(1), singularLimits, ...
    [char(215) '10' char(8315) char(179) ' sp/s']);
singularColorbar.Ruler.Exponent = 0;
singularColorbar.TickLabels = compose('%g', singularColorbar.Ticks * 1e3);
local_layout_text(fig, distributionAxes, ...
    [fixedColorbar eigenColorbar singularColorbar], lefts, bottoms, height);

if strlength(string(outputFig)) > 0
    savefig(fig, outputFig);
end
fig.PaperUnits = 'inches';
fig.PaperSize = fig.Position(3:4);
fig.PaperPosition = [0 0 fig.Position(3:4)];
print(fig, outputPdf, '-dpdf', '-painters', '-r600');
close(fig);
fprintf('Saved %s\n', outputPdf);
end

function ax = local_heatmap(fig, position, values, limits, showXLabel)
ax = axes(fig, 'Units', 'inches', 'Position', position);
imagesc(ax, values);
axis(ax, 'image');
set(ax, 'YDir', 'normal');
colormap(ax, jet(256));
caxis(ax, limits);
xlim(ax, [0.5 40.5]);
ylim(ax, [0.5 40.5]);
xticks(ax, []);
yticks(ax, []);
xticklabels(ax, {});
yticklabels(ax, {});
set(ax, 'FontName', 'Arial', 'FontSize', 35, 'LineWidth', 1.25, ...
    'Box', 'on', 'TickDir', 'out', 'Layer', 'top', 'TickLabelInterpreter', 'none');
ax.Toolbar.Visible = 'off';
disableDefaultInteractivity(ax);
hold(ax, 'on');
for boundary = [10.5 20.5 30.5]
    xline(ax, boundary, '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.8);
    yline(ax, boundary, '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.8);
end
hold(ax, 'off');
end

function ax = local_realpart_distribution(fig, position, values, showXLabel)
ax = axes(fig, 'Units', 'inches', 'Position', position);
edges = -2.1:0.05:1.1;
counts = histcounts(real(values), edges);
centers = edges(1:end-1) + diff(edges) / 2;
bar(ax, centers, counts, 1, 'FaceColor', [0.78 0.78 0.78], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.65);
if showXLabel
    ylabel(ax, 'Count', 'Interpreter', 'none');
    xlabel(ax, 'Real part of eigenvalue', 'Interpreter', 'none');
end
xlim(ax, [-2.1 1.1]);
ylim(ax, [0, 1.08 * max(counts)]);
hold(ax, 'on');
xline(ax, 0, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.6);
xline(ax, 1, '--', 'Color', [178 24 43] / 255, 'LineWidth', 1.5);
hold(ax, 'off');
axis(ax, 'square');
set(ax, 'FontName', 'Arial', 'FontSize', 35, 'LineWidth', 1.25, ...
    'Box', 'on', 'TickDir', 'out', 'Layer', 'top', 'TickLabelInterpreter', 'none');
ax.Toolbar.Visible = 'off';
disableDefaultInteractivity(ax);
ax.XLabel.FontSize = 45;
ax.YLabel.FontSize = 45;
ax.XLabel.FontWeight = 'bold';
ax.YLabel.FontWeight = 'bold';
end

function cb = local_colorbar(ax, limits, unitText)
axisPosition = ax.Position;
cb = colorbar(ax, 'eastoutside');
ax.Position = axisPosition;
cb.Units = 'inches';
cb.Position = [axisPosition(1) + axisPosition(3) + 0.3, ...
    axisPosition(2), 30/72, axisPosition(4)];
cb.Limits = limits;
cb.FontName = 'Arial';
cb.FontSize = 35;
cb.FontWeight = 'normal';
cb.TickLabelInterpreter = 'none';
cb.LineWidth = 1.1;
cb.Title.String = unitText;
cb.Title.Interpreter = 'none';
cb.Title.FontName = 'Arial';
cb.Title.FontSize = 45;
cb.Title.FontWeight = 'bold';
cb.Title.Units = 'normalized';
cb.Title.Position = [0.5 1 + 7/(72*axisPosition(4)) 0];
cb.Title.VerticalAlignment = 'bottom';
end

function local_layout_text(fig, distributionAxes, colorbars, lefts, bottoms, side)
% Explicit baselines avoid incorrect headless MATLAB text alignment metrics.
fontDir = fullfile(fileparts(mfilename('fullpath')), 'fonts');
if ismac
    fontDir = '/System/Library/Fonts/Supplemental';
end
regular = java.awt.Font.createFont(0, java.io.File(fullfile(fontDir, 'Arial.ttf')));
bold = java.awt.Font.createFont(0, java.io.File(fullfile(fontDir, 'Arial Bold.ttf')));
regular = regular.deriveFont(single(35));
bold = bold.deriveFont(single(45));
frc = java.awt.font.FontRenderContext(java.awt.geom.AffineTransform(), true, true);
rm = regular.getLineMetrics('Hg', frc);
bm = bold.getLineMetrics('Hg', frc);
ra = double(rm.getAscent())/72;
rd = double(rm.getDescent())/72;
ba = double(bm.getAscent())/72;
bd = double(bm.getDescent())/72;
drawnow;
canvas = fig.Position(3:4);
overlay = axes(fig, 'Units', 'inches', 'Position', [0 0 canvas], ...
    'XLim', [0 canvas(1)], 'YLim', [0 canvas(2)], 'Visible', 'off', ...
    'Color', 'none', 'Tag', 'LayoutText', 'HitTest', 'off');
titles = {{'E Firing Rate'}, {'Eigenvalue Realpart', 'Distribution'}, ...
    {'First Eigenvector'}, {'Top Singular Mode'}};
for col = 1:4
    for line = 1:numel(titles{col})
        str = titles{col}{line};
        x = lefts(col) + side/2 - local_text_width(bold, frc, str)/2;
        local_text(overlay, x, bottoms(1)+side+1.2-(line-1)*0.95, str, 'bold', 0);
    end
end
for row = 1:2
    ax = distributionAxes(row);
    p = ax.Position;
    xTicks = ax.XTick;
    yTicks = ax.YTick;
    ax.XTickLabel = {};
    ax.YTickLabel = {};
    ax.XLabel.Visible = 'off';
    ax.YLabel.Visible = 'off';
    xBase = p(2)-0.1-ra;
    for tick = xTicks
        str = sprintf('%g', tick);
        x = p(1)+(tick-ax.XLim(1))/diff(ax.XLim)*p(3);
        local_text(overlay, x-local_text_width(regular, frc, str)/2, xBase, str, 'normal', 0);
    end
    tickLeft = p(1);
    for tick = yTicks
        str = sprintf('%g', tick);
        x = p(1)-0.06-local_text_width(regular, frc, str);
        y = p(2)+(tick-ax.YLim(1))/diff(ax.YLim)*p(4)-(ra-rd)/2;
        local_text(overlay, x, y, str, 'normal', 0);
        tickLeft = min(tickLeft, x);
    end
    if row == 2
        str = 'Real part of eigenvalue';
        local_text(overlay, p(1)+p(3)/2-local_text_width(bold, frc, str)/2, ...
            xBase-rd-0.4-ba, str, 'bold', 0);
        str = 'Count';
        local_text(overlay, tickLeft-0.4-bd, ...
            p(2)+p(4)/2-local_text_width(bold, frc, str)/2, str, 'bold', 90);
    end
end
for k = 1:numel(colorbars)
    cb = colorbars(k);
    p = cb.Position;
    ticks = cb.Ticks;
    cb.TickLabels = repmat({''}, size(ticks));
    cb.Title.Visible = 'off';
    for tick = ticks
        value = tick;
        if k == 3
            value = value*1e3;
        end
        str = sprintf('%g', value);
        y = p(2)+(tick-cb.Limits(1))/diff(cb.Limits)*p(4)-(ra-rd)/2;
        local_text(overlay, p(1)+p(3)+0.06, y, str, 'normal', 0);
    end
    str = cb.Title.String;
    local_text(overlay, p(1)+p(3)/2-local_text_width(bold, frc, str)/2, ...
        p(2)+p(4)+0.1+bd, str, 'bold', 0);
end
end

function width = local_text_width(font, frc, str)
bounds = font.getStringBounds(str, frc);
width = double(bounds.getWidth())/72;
end

function local_text(ax, x, y, str, weight, rotation)
fontSize = 45;
if strcmp(weight, 'normal')
    fontSize = 35;
end
text(ax, x, y, str, 'FontName', 'Arial', 'FontSize', fontSize, ...
    'FontWeight', weight, 'Interpreter', 'none', 'Rotation', rotation, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'baseline', ...
    'Clipping', 'off');
end
