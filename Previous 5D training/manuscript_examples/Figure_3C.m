function Figure_3C(projectRoot)
% Combined Figure 4C retains the historical standalone Figure 3C filenames.
% Each row shows its E-to-E kernel, real-part distribution, and Fourier gain.

if nargin == 0
    projectRoot = '/Users/xialeihuang/Desktop/Neuroscience_Project';
end
analysisRoot = fullfile(projectRoot, 'Spectral_Analysis', ...
    'matlab-inserting_into_CG_model', 'l6_next_stage_analysis_20260717');
controlRoot = fullfile(analysisRoot, ...
    'spatial_symmetry_zero_controls_20260811');
gkeRoot = fullfile(analysisRoot, 'gaussian_kernel_experiments_20260813');
resultRoot = fullfile(controlRoot, 'reequilibrated_all_controls_results');
setupFile = fullfile(analysisRoot, 'results_global_bifurcation_20260722', ...
    'global_bifurcation_setup.mat');
runtimeRoot = fullfile(projectRoot, 'Spectral_Analysis', ...
    'matlab-inserting_into_CG_model', 'geometry_analysis', ...
    'torch_h96_dedupe_array', 'Utils');
utilsRoot = fullfile(projectRoot, 'Spectral_Analysis', ...
    'matlab-inserting_into_CG_model', 'Complete_Code_for_Paper3', ...
    'NYU-Vision-2Drive-main', 'Utils');
outputDir = fileparts(mfilename('fullpath'));
outputPdf = fullfile(outputDir, 'Figure 3C.pdf');
outputFig = fullfile(outputDir, 'Figure 3C.fig');

addpath(analysisRoot);
addpath(controlRoot);
addpath(gkeRoot);
addpath(utilsRoot);
addpath(runtimeRoot, '-begin');

setupData = load(setupFile, 'setup');
setup = setupData.setup;
extraSpectrumRoot = fullfile(projectRoot, 'Spectral_Analysis', ...
    'matlab-inserting_into_CG_model', 'report_figure_artifacts', ...
    '3_Weighted_Jacobian', 'Figure_3C', 'gke_cache', 'complete_spectra');

saved = {
    load(fullfile(resultRoot, 'baseline.mat'))
    load(fullfile(resultRoot, 'remove_L4_smoothing.mat'))
    load(fullfile(extraSpectrumRoot, 'kernel_noise_cv0p30.mat'))
    load(fullfile(extraSpectrumRoot, 'shape_left_half.mat'))
};
operators = cell(4, 1);
[operators{1}, ~] = l6ns_full_spatial_control_operators( ...
    setup.Context, 'baseline');
[operators{2}, ~] = l6ns_full_spatial_control_operators( ...
    setup.Context, 'remove_L4_smoothing');
[operators{3}, ~] = gke_build_operators( ...
    setup.Context, 'kernel_noise_cv0p30');
[operators{4}, ~] = gke_build_operators( ...
    setup.Context, 'shape_left_half');

rowOrder = [1 3 4 2];
saved = saved(rowOrder);
operators = operators(rowOrder);
labels = {'Baseline', 'Noised (CV 0.30)', 'Left Half', 'Flattened'};
numberOfSystems = numel(labels);
for index = 1:numberOfSystems
    systems(index).Label = labels{index}; %#ok<AGROW>
    systems(index).KernelEE = local_centered_kernel( ...
        operators{index}.C_SS, [40 40]); %#ok<AGROW>
    systems(index).Eigenvalues = saved{index}.eigenvalues(:); %#ok<AGROW>
    assert(numel(systems(index).Eigenvalues) == 4800, ...
        'Each spectrum must contain 4,800 eigenvalues.');
end

% Match Figure 3B: linear pair probability on its baseline support window.
baselineKernel = systems(1).KernelEE;
supportRows = find(any(baselineKernel ~= 0, 2));
supportColumns = find(any(baselineKernel ~= 0, 1));
kernelOffset = -20:19;
kernelDisplayX = kernelOffset(supportColumns);
kernelDisplayY = kernelOffset(supportRows);
kernelLimits = [0 max(baselineKernel, [], 'all')];
allEigenvalues = vertcat(systems.Eigenvalues);
binWidth = 0.1;
firstEdgeIndex = floor((min(real(allEigenvalues))+0.05)/binWidth);
lastEdgeIndex = ceil((max(1.08, max(real(allEigenvalues)))+0.05)/binWidth);
binEdges = (firstEdgeIndex:lastEdgeIndex)*binWidth-0.05;
xLimits = binEdges([1 end]);
histogramMaximum = 0;
for index = 1:numberOfSystems
    systems(index).HistogramCounts = histcounts( ...
        real(systems(index).Eigenvalues), binEdges);
    assert(sum(systems(index).HistogramCounts) == 4800, ...
        'Histogram counts must sum to 4,800.');
    histogramMaximum = max(histogramMaximum, ...
        max(systems(index).HistogramCounts));
end

nearZeroBins = binEdges(1:end-1) >= -0.05-1e-12 & ...
    binEdges(2:end) <= 0.05+1e-12;
assert(nnz(nearZeroBins) == 1);
assert(max(abs([binEdges(nearZeroBins) binEdges(find(nearZeroBins)+1)]-[-0.05 0.05])) < 1e-12);

% Match Figure 3B's full-field FFT before cropping the displayed kernel.
n = size(baselineKernel, 1);
weakThreshold = 0.05;
gainHistogramMaximum = 0;
for index = 1:numberOfSystems
    kernelNormalized = systems(index).KernelEE / ...
        sum(systems(index).KernelEE, 'all');
    transferComplex = fft2(ifftshift(kernelNormalized));
    transferGain = abs(fftshift(transferComplex));
    systems(index).FourierGain = transferGain / max(transferGain, [], 'all');
    assert(numel(systems(index).FourierGain) == 1600);
    assert(abs(systems(index).FourierGain(n/2+1,n/2+1)-1) < 1e-12);
    gain = systems(index).FourierGain(:);
    if index == 1
        % Preserve the original Figure 3B gain-axis limits and logarithmic scale.
        gainLimits = [max(min(gain)/2, 1e-7) 1.25];
        gainBinEdges = unique([logspace(log10(gainLimits(1)), ...
            log10(gainLimits(2)), 36) weakThreshold]);
    end
    assert(all(gain >= gainLimits(1) & gain <= gainLimits(2)));
    weakGain = gain <= weakThreshold;
    systems(index).WeakGainCounts = histcounts(gain(weakGain), gainBinEdges);
    systems(index).StrongGainCounts = histcounts(gain(~weakGain), gainBinEdges);
    assert(sum(systems(index).WeakGainCounts+systems(index).StrongGainCounts) == 1600);
    gainHistogramMaximum = max(gainHistogramMaximum, ...
        max(systems(index).WeakGainCounts+systems(index).StrongGainCounts));
end

neuroColors = [
    178  24  43
    227  74  51
    254 227 145
    171 217 233
     67 147 195
     33 102 172
] / 255;

fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
    'Position', [0.2 0.2 41.8 38.3], 'PaperPositionMode', 'auto');
fig.Position = [0.2 0.2 41.8 38.3];
columnX = [1.2 12.2 30.8];
rowY = [29.1 20.0 10.9 1.8];
panelSize = 7.6;
distributionWidth = 2*panelSize;
plotAxes = gobjects(numberOfSystems, 3);

for row = 1:numberOfSystems
    systemIndex = row;
    kernelPosition = [columnX(1) rowY(row) panelSize panelSize];
    kernelAx = axes(fig, 'Units', 'inches', 'Position', kernelPosition, ...
        'PositionConstraint', 'innerposition');
    kernelDisplay = systems(systemIndex).KernelEE(supportRows, supportColumns);
    imagesc(kernelAx, kernelDisplayX, kernelDisplayY, kernelDisplay);
    axis(kernelAx, 'image');
    xlim(kernelAx, [kernelDisplayX(1)-0.5 kernelDisplayX(end)+0.5]);
    ylim(kernelAx, [kernelDisplayY(1)-0.5 kernelDisplayY(end)+0.5]);
    set(kernelAx, 'YDir', 'normal', 'XTick', [], 'YTick', [], ...
        'FontName', 'Arial', 'FontSize', 41, 'FontWeight', 'normal', ...
        'LineWidth', 1.2, 'Box', 'on', 'CLim', kernelLimits);
    disableDefaultInteractivity(kernelAx);
    kernelAx.Toolbar.Visible = 'off';
    colormap(kernelAx, jet(256));
    if row == 1
        title(kernelAx, 'E-to-E Kernel', 'FontName', 'Arial', ...
            'FontSize', 51, 'FontWeight', 'bold');
    end
    fprintf('%s kernel: peak %.6g; %d pixels above shared color limit %.6g.\n', ...
        systems(systemIndex).Label, max(kernelDisplay, [], 'all'), ...
        nnz(kernelDisplay > kernelLimits(2)), kernelLimits(2));
    if row == 1
        cb = colorbar(kernelAx);
        cb.Units = 'inches';
        cb.Position = [columnX(1)+panelSize+0.5 rowY(1) 40/72 panelSize];
        cb.FontName = 'Arial';
        cb.FontSize = 41;
        cb.FontWeight = 'normal';
        cb.LineWidth = 1.0;
        cb.Title.String = 'P_{pair}';
        cb.Title.Interpreter = 'tex';
        cb.Title.FontSize = 51;
        cb.Title.FontWeight = 'bold';
        cb.Title.Units = 'normalized';
        cb.Title.Position = [0.5 1+0.1/panelSize 0];
        cb.Title.HorizontalAlignment = 'center';
        cb.Title.VerticalAlignment = 'bottom';
        kernelAx.Position = kernelPosition;
    end

    distributionAx = axes(fig, 'Units', 'inches', ...
        'Position', [columnX(2) rowY(row) distributionWidth panelSize], ...
        'PositionConstraint', 'innerposition');
    local_realpart_histogram(distributionAx, binEdges, ...
        systems(systemIndex).HistogramCounts, xLimits, ...
        histogramMaximum, nearZeroBins, neuroColors);
    if strcmp(systems(systemIndex).Label, 'Flattened')
        yline(distributionAx, 1600, '--', 'Color', [0.5 0.5 0.5], ...
            'LineWidth', 2.5, 'HandleVisibility', 'off');
    end
    if row < numberOfSystems
        distributionAx.XLabel.Visible = 'off';
    end
    disableDefaultInteractivity(distributionAx);
    distributionAx.Toolbar.Visible = 'off';
    if row == 1
        title(distributionAx, 'Real-part Eigenvalue Distribution', ...
            'FontName', 'Arial', 'FontSize', 51, 'FontWeight', 'bold');
    end
    gainAx = axes(fig, 'Units', 'inches', ...
        'Position', [columnX(3) rowY(row) panelSize panelSize], ...
        'PositionConstraint', 'innerposition');
    local_fourier_gain(gainAx, gainBinEdges, ...
        systems(systemIndex).WeakGainCounts, systems(systemIndex).StrongGainCounts, ...
        gainHistogramMaximum, neuroColors);
    if row < numberOfSystems
        ylim(gainAx, [0 500]);
        gainAx.XLabel.Visible = 'off';
    end
    if row == 1
        title(gainAx, 'Gain Distribution', 'FontName', 'Arial', ...
            'FontSize', 51, 'FontWeight', 'bold');
    end
    plotAxes(row, :) = [kernelAx distributionAx gainAx];
    fprintf('%s: histogram %d; near-zero bins %d; Fourier modes %d; gains <= 0.05: %d.\n', ...
        systems(systemIndex).Label, sum(systems(systemIndex).HistogramCounts), ...
        sum(systems(systemIndex).HistogramCounts(nearZeroBins)), ...
        numel(systems(systemIndex).FourierGain), ...
        nnz(systems(systemIndex).FourierGain <= weakThreshold));
end
drawnow;
% Preserve the physical plot boxes and 0.4-inch axis-title gaps.
layoutAxes = plotAxes(:);
for layoutIndex = 1:numel(layoutAxes)
    layoutAx = layoutAxes(layoutIndex);
    assert(abs(layoutAx.Position(4)-panelSize) < 1e-8);
    xText = layoutAx.XLabel.String;
    yText = layoutAx.YLabel.String;
    xVisible = layoutAx.XLabel.Visible;
    yVisible = layoutAx.YLabel.Visible;
    set([layoutAx.XLabel layoutAx.YLabel], 'Visible', 'off');
    drawnow;
    tickInset = layoutAx.TightInset;
    layoutAx.XLabel.Visible = xVisible;
    layoutAx.YLabel.Visible = yVisible;
    set([layoutAx.XLabel layoutAx.YLabel], 'Units', 'inches');
    drawnow;
    if ~isempty(xText) && strcmp(xVisible, 'on')
        extent = layoutAx.XLabel.Extent;
        position = layoutAx.XLabel.Position;
        position(2) = position(2)-tickInset(2)-0.4-extent(2)-extent(4);
        layoutAx.XLabel.Position = position;
    end
    if ~isempty(yText) && strcmp(yVisible, 'on')
        extent = layoutAx.YLabel.Extent;
        position = layoutAx.YLabel.Position;
        if strcmp(layoutAx.YAxisLocation, 'right')
            position(1) = position(1)+layoutAx.Position(3)+tickInset(3)+0.4-extent(1);
        else
            position(1) = position(1)-tickInset(1)-0.4-extent(1)-extent(3);
        end
        layoutAx.YLabel.Position = position;
    end
end
% Place shared colorbar text and case names independently of the plot boxes.
cbPosition = cb.Position;
cbTicks = cb.Ticks;
cb.Ticks = [];
cb.Title.Visible = 'off';
canvas = fig.Position(3:4);
textAxes = axes(fig, 'Units', 'inches', 'Position', [0 0 canvas], ...
    'XLim', [0 canvas(1)], 'YLim', [0 canvas(2)], ...
    'Visible', 'off', 'Color', 'none', 'Tag', 'ColorbarText');
for tick = cbTicks
    tickY = cbPosition(2)+(tick-cb.Limits(1))/diff(cb.Limits)*cbPosition(4);
    text(textAxes, cbPosition(1)+cbPosition(3)+0.06, tickY, sprintf('%g', tick), ...
        'FontName', 'Arial', 'FontSize', 41, 'FontWeight', 'normal', ...
        'Interpreter', 'none', 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', 'Clipping', 'off');
end
text(textAxes, cbPosition(1)+cbPosition(3)/2, cbPosition(2)+cbPosition(4)+0.1, ...
    cb.Title.String, 'FontName', 'Arial', 'FontSize', 51, 'FontWeight', 'bold', ...
    'Interpreter', 'tex', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'Clipping', 'off');
for row = 1:numberOfSystems
    text(textAxes, 0.40, rowY(row)+panelSize/2, ...
        strrep(labels{row}, '0.30', '{\rm0.30}'), 'Rotation', 90, ...
        'FontName', 'Arial', 'FontSize', 51, 'FontWeight', 'bold', ...
        'Interpreter', 'tex', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'Clipping', 'off');
    assert(max(abs(plotAxes(row, 1).Position-[columnX(1) rowY(row) panelSize panelSize])) < 1e-8);
    assert(max(abs(plotAxes(row, 2).Position-[columnX(2) rowY(row) distributionWidth panelSize])) < 1e-8);
    assert(max(abs(plotAxes(row, 3).Position-[columnX(3) rowY(row) panelSize panelSize])) < 1e-8);
end
fig.UserData = struct('CombinedPanel', '4C', 'Cases', {labels}, ...
    'BinEdges', binEdges, 'NearZeroBins', nearZeroBins, ...
    'GainBinEdges', gainBinEdges, 'GainLimits', gainLimits, ...
    'WeakGainThreshold', weakThreshold, 'Systems', systems);
drawnow;
savefig(fig, outputFig);
builtFigure = fig;
% Reopen the saved scene to avoid stale text textures in the macOS PDF renderer.
fig = openfig(outputFig, 'invisible');
fig.Units = 'inches';
fig.Position = [0.2 0.2 41.8 38.3];
drawnow;
exportgraphics(fig, outputPdf, 'ContentType', 'vector', 'Padding', 16);
close(fig);
close(builtFigure);
fprintf('Saved %s\n', outputPdf);
end

function kernel = local_centered_kernel(matrix, mapSize)
pixelsPerHC = 10;
sourceRow = floor(mapSize(1)/2)-floor(pixelsPerHC/2);
sourceColumn = floor(mapSize(2)/2)-floor(pixelsPerHC/2);
sourceIndex = sourceRow+(sourceColumn-1)*mapSize(1);
neuronsPerPixel = 45^2/pixelsPerHC^2;
kernel = reshape(full(matrix(sourceIndex, :)), mapSize)/neuronsPerPixel;
kernel = circshift(kernel, floor(mapSize/2)+1-[sourceRow sourceColumn]);
end

function local_realpart_histogram(ax, binEdges, counts, xLimits, ...
        histogramMaximum, nearZeroBins, neuroColors)
binCenters = binEdges(1:end-1) + diff(binEdges) / 2;
hold(ax, 'on');
bars = bar(ax, binCenters, counts, 1, 'BaseValue', 0, ...
    'FaceColor', 'flat', ...
    'EdgeColor', 'none', 'HandleVisibility', 'off');
bars.CData = repmat([0.62 0.64 0.67], numel(counts), 1);
bars.CData(nearZeroBins, :) = repmat(neuroColors(5, :), nnz(nearZeroBins), 1);
xline(ax, 0, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.8, ...
    'HandleVisibility', 'off');
xline(ax, 1, '--', 'Color', neuroColors(1, :), 'LineWidth', 2.2, ...
    'HandleVisibility', 'off');
xlabel(ax, 'Real part of eigenvalue', 'FontSize', 51, 'FontWeight', 'bold');
xlim(ax, xLimits);
ylim(ax, [0 1.10 * histogramMaximum]);
yticks(ax, 0:500:1.10*histogramMaximum);
set(ax, 'FontName', 'Arial', 'FontSize', 41, 'FontWeight', 'normal', ...
    'LineWidth', 1.2, 'Box', 'on', ...
    'Layer', 'top', 'Color', 'w', 'XGrid', 'off', 'YGrid', 'off', ...
    'YAxisLocation', 'right');
set([ax.XLabel ax.YLabel], 'FontSize', 51, 'FontWeight', 'bold');
end

function local_fourier_gain(ax, binEdges, weakCounts, strongCounts, ...
        histogramMaximum, neuroColors)
hold(ax, 'on');
% Explicit edges preserve logarithmic bin widths and the 0.05 color boundary.
histogram(ax, 'BinEdges', binEdges, 'BinCounts', weakCounts, ...
    'FaceColor', neuroColors(5, :), 'FaceAlpha', 1, ...
    'EdgeColor', 'none', 'HandleVisibility', 'off');
histogram(ax, 'BinEdges', binEdges, 'BinCounts', strongCounts, ...
    'FaceColor', neuroColors(1, :), 'FaceAlpha', 1, ...
    'EdgeColor', 'none', 'HandleVisibility', 'off');
xlabel(ax, 'Gain', ...
    'FontSize', 51, 'FontWeight', 'bold');
ylabel(ax, 'Count', 'FontSize', 51, 'FontWeight', 'bold');
set(ax, 'XScale', 'log', 'XLim', binEdges([1 end]), ...
    'XTick', 10.^(ceil(log10(binEdges(1))):floor(log10(binEdges(end)))), ...
    'XTickLabel', {'10^{-5}', '', '10^{-3}', '', '10^{-1}', '10^{0}'}, ...
    'XTickLabelRotation', 0, ...
    'YLim', [0 1.1*histogramMaximum], ...
    'YAxisLocation', 'right', 'FontName', 'Arial', 'FontSize', 41, ...
    'FontWeight', 'normal', 'LineWidth', 1.4, 'Box', 'on', 'Layer', 'top', ...
    'TickDir', 'out', 'XColor', [0.18 0.18 0.18], 'YColor', [0.18 0.18 0.18]);
set([ax.XLabel ax.YLabel], 'FontSize', 51, 'FontWeight', 'bold');
disableDefaultInteractivity(ax);
ax.Toolbar.Visible = 'off';
assert(sum(weakCounts+strongCounts) == 1600);
end
