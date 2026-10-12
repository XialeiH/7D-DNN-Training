function plot_sftf_tuning_curves(root, pixels)
% Plot established h96 7D TF/SF equilibrium tuning for one or more pixels.
if nargin < 1 || isempty(root)
    root = getenv('SFTF_TUNING_ROOT');
end
if isempty(root)
    error('Provide the tuning root or set SFTF_TUNING_ROOT.');
end
if nargin < 2 || isempty(pixels)
    pixels = [5 10];
end
if size(pixels, 2) ~= 2
    error('pixels must be an N-by-2 matrix of MATLAB [row column] coordinates.');
end

outDir = fullfile(root, 'plots');
if ~exist(outDir, 'dir'), mkdir(outDir); end
tfValues = [0.5 1 2 4 10 16 24 32];
sfValues = [0.5 1 2 2.5 3 5 10];
allTfRates = cell(size(pixels, 1), 1);
allSfRates = cell(size(pixels, 1), 1);

for pixelId = 1:size(pixels, 1)
    row = pixels(pixelId, 1);
    col = pixels(pixelId, 2);
    tfRates = readRates(root, tfValues, 2.5, row, col);
    sfRates = readRates(root, 10, sfValues, row, col);
    allTfRates{pixelId} = tfRates;
    allSfRates{pixelId} = sfRates;
    suffix = sprintf('pixel%d_%d', row, col);

    writetable(array2table([tfValues(:), tfRates], ...
        'VariableNames', {'TF', 'S', 'C', 'I', 'E'}), ...
        fullfile(outDir, ['tf_tuning_angle0_contr100_' suffix '.csv']));
    writetable(array2table([sfValues(:), sfRates], ...
        'VariableNames', {'SF', 'S', 'C', 'I', 'E'}), ...
        fullfile(outDir, ['sf_tuning_angle0_contr100_' suffix '.csv']));
    save(fullfile(outDir, ['tfsf_tuning_angle0_contr100_' suffix '.mat']), ...
        'tfValues', 'sfValues', 'tfRates', 'sfRates', 'row', 'col');
    makeSingleFigure(tfValues, tfRates, 'Temporal frequency TF', ...
        sprintf('TF tuning | angle 0.0, contrast 100, SF 2.5, pixel (%d,%d)', row, col), ...
        fullfile(outDir, ['tf_tuning_angle0_contr100_' suffix]));
    makeSingleFigure(sfValues, sfRates, 'Spatial frequency SF', ...
        sprintf('SF tuning | angle 0.0, contrast 100, TF 10, pixel (%d,%d)', row, col), ...
        fullfile(outDir, ['sf_tuning_angle0_contr100_' suffix]));
end

if size(pixels, 1) > 1
    makeComparisonFigure(outDir, pixels, tfValues, sfValues, allTfRates, allSfRates);
end
end


function rates = readRates(root, tfValues, sfValues, row, col)
[tfGrid, sfGrid] = ndgrid(tfValues, sfValues);
rates = nan(numel(tfGrid), 4);
for idx = 1:numel(tfGrid)
    conditionDir = fullfile(root, 'equilibrium', sprintf('TF%s_SF%s', ...
        numLabel(tfGrid(idx)), numLabel(sfGrid(idx))));
    path = fullfile(conditionDir, ...
        'eig_16HC_6DprefAngle_input_contr100_angle_0.00.mat');
    if ~isfile(path), error('Missing NN equilibrium: %s', path); end
    D = load(path, 'S_map_full', 'C_map_full', 'I_map_full');
    if row > size(D.S_map_full, 1) || col > size(D.S_map_full, 2)
        error('Pixel (%d,%d) exceeds map size in %s.', row, col, path);
    end
    S = D.S_map_full(row, col);
    C = D.C_map_full(row, col);
    I = D.I_map_full(row, col);
    E = (1 - 0.3077) * S + 0.3077 * C;
    rates(idx,:) = [S C I E];
end
end


function makeSingleFigure(x, rates, xLabel, titleText, stem)
names = {'S', 'C', 'I', 'E'};
colors = channelColors();
fig = figure('Color', 'w', 'Position', [100 100 1180 840]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
for channel = 1:4
    ax = nexttile(layout);
    semilogx(ax, x, rates(:,channel), '-o', 'LineWidth', 2.2, ...
        'MarkerSize', 7, 'Color', colors(channel,:), 'MarkerFaceColor', colors(channel,:));
    formatAxes(ax, x, xLabel);
    ylabel(ax, 'firing rate (Hz)');
    title(ax, sprintf('%s firing rate', names{channel}), 'FontWeight', 'bold');
end
title(layout, titleText, 'FontWeight', 'bold', 'FontSize', 14);
saveFigure(fig, stem);
end


function makeComparisonFigure(outDir, pixels, tfValues, sfValues, tfRates, sfRates)
names = {'S', 'C', 'I', 'E'};
colors = lines(size(pixels, 1));
fig = figure('Color', 'w', 'Position', [80 80 1720 850]);
layout = tiledlayout(fig, 2, 4, 'TileSpacing', 'compact', 'Padding', 'loose');
for channel = 1:4
    ax = nexttile(layout, channel);
    hold(ax, 'on');
    for pixelId = 1:size(pixels, 1)
        semilogx(ax, tfValues, tfRates{pixelId}(:,channel), '-o', ...
            'LineWidth', 2, 'Color', colors(pixelId,:), 'MarkerSize', 6);
    end
    formatAxes(ax, tfValues, 'Temporal frequency TF');
    ylabel(ax, 'firing rate (Hz)');
    title(ax, sprintf('%s: TF tuning at SF 2.5', names{channel}), 'FontWeight', 'bold');

    ax = nexttile(layout, 4 + channel);
    hold(ax, 'on');
    for pixelId = 1:size(pixels, 1)
        semilogx(ax, sfValues, sfRates{pixelId}(:,channel), '-o', ...
            'LineWidth', 2, 'Color', colors(pixelId,:), 'MarkerSize', 6);
    end
    formatAxes(ax, sfValues, 'Spatial frequency SF');
    ylabel(ax, 'firing rate (Hz)');
    title(ax, sprintf('%s: SF tuning at TF 10', names{channel}), 'FontWeight', 'bold');
end
labels = arrayfun(@(id) sprintf('pixel (%d,%d)', pixels(id,1), pixels(id,2)), ...
    1:size(pixels,1), 'UniformOutput', false);
legend(layout.Children(end), labels, 'Location', 'northwest', 'Box', 'off');
title(layout, '7D NN TF/SF tuning comparison | angle 0.0, contrast 100', ...
    'FontWeight', 'bold', 'FontSize', 15);
saveFigure(fig, fullfile(outDir, 'sftf_tuning_pixel_comparison'));
end


function formatAxes(ax, x, xLabel)
grid(ax, 'on');
box(ax, 'off');
set(ax, 'XTick', x, 'XTickLabel', compose('%g', x), ...
    'FontName', 'Helvetica', 'FontSize', 10, 'LineWidth', 1);
xlim(ax, [min(x)*0.88 max(x)*1.12]);
xlabel(ax, xLabel);
end


function saveFigure(fig, stem)
exportgraphics(fig, [stem '.png'], 'Resolution', 220);
exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector');
savefig(fig, [stem '.fig']);
close(fig);
end


function colors = channelColors()
colors = [0.06 0.36 0.55; 0.89 0.36 0.13; 0.15 0.55 0.38; 0.72 0.55 0.12];
end


function label = numLabel(value)
label = regexprep(regexprep(sprintf('%.4f', value), '0+$', ''), '\.$', '');
label = strrep(label, '.', 'p');
end
