function plot_h96_7d_orientation_contrast(atlasRoot, outRoot)
% Plot default h96 orientation and contrast curves from a complete equ102 atlas.
if nargin < 1 || isempty(atlasRoot), atlasRoot = getenv('H96_EQU102_ROOT'); end
if nargin < 2 || isempty(outRoot), outRoot = fullfile(atlasRoot, 'tuning_curves'); end
if isempty(atlasRoot), error('Provide atlasRoot or set H96_EQU102_ROOT.'); end
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

T = readtable(fullfile(atlasRoot, 'cases.tsv'), 'FileType', 'text', ...
    'Delimiter', '\t', 'VariableNamingRule', 'preserve');
T = T(T.case <= 70, :);
angles = unique(T.angle, 'sorted')';
contrasts = unique(T.contrast, 'sorted')';
if height(T) ~= 70 || numel(angles) ~= 7 || numel(contrasts) ~= 10
    error('Expected the complete 7-angle x 10-contrast equ70 subset.');
end
[S, C, I] = loadMaps(T, angles, contrasts);
E = (1 - 0.3077) .* S + 0.3077 .* C;
channels = {S, C, I, E};
names = {'S', 'C', 'I', 'E'};
orientationContrast = max(contrasts);
plotOrientation(outRoot, channels, names, angles, contrasts, orientationContrast);
plotContrast(outRoot, channels, names, angles, contrasts, 0, 5, 10, 'angle0');
plotContrast(outRoot, channels, names, angles, contrasts, 22.5, 1, 10, 'angle22p5');
save(fullfile(outRoot, 'orientation_contrast_tuning_data.mat'), ...
    'S', 'C', 'I', 'E', 'angles', 'contrasts', 'orientationContrast');
end


function [S, C, I] = loadMaps(T, angles, contrasts)
S = []; C = []; I = [];
for rowId = 1:height(T)
    saveDir = char(string(T.save_dir(rowId)));
    files = dir(fullfile(saveDir, 'eig_16HC_6DprefAngle_input_contr*_angle_*.mat'));
    if numel(files) ~= 1, error('Expected one equilibrium MAT in %s.', saveDir); end
    D = load(fullfile(files(1).folder, files(1).name), ...
        'S_map_full', 'C_map_full', 'I_map_full');
    if isempty(S)
        sz = size(D.S_map_full);
        S = nan(numel(angles), numel(contrasts), sz(1), sz(2));
        C = S; I = S;
    end
    ai = find(abs(angles - T.angle(rowId)) < 1e-8, 1);
    ci = find(abs(contrasts - T.contrast(rowId)) < 1e-8, 1);
    S(ai,ci,:,:) = D.S_map_full;
    C(ai,ci,:,:) = D.C_map_full;
    I(ai,ci,:,:) = D.I_map_full;
end
if any(~isfinite(S(:))) || any(~isfinite(C(:))) || any(~isfinite(I(:)))
    error('The equ70 maps contain missing or nonfinite values.');
end
end


function plotOrientation(outRoot, channels, names, angles, contrasts, contrastUse)
fullAngles = 0:3.75:180;
rows = [5 1 3]; cols = [10 10 9];
labels = {'pixel angle 0', 'pixel angle 22.5', 'middle pixel'};
rates = nan(numel(fullAngles), 3, 4);
[~, ci] = min(abs(contrasts - contrastUse));
for pixelId = 1:3
    for angleId = 1:numel(fullAngles)
        if pixelId < 3
            [canon, row, col] = mapDefaultPixel(fullAngles(angleId), rows(pixelId), cols(pixelId));
        else
            [canon, row, col] = mapMiddlePixel(fullAngles(angleId));
        end
        ai = find(abs(angles - canon) < 1e-8, 1);
        if isempty(ai), error('Missing canonical angle %.2f.', canon); end
        for channelId = 1:4
            X = channels{channelId};
            rates(angleId,pixelId,channelId) = X(ai,ci,row,col);
        end
    end
end

wrapped = fullAngles >= 157.5;
order = [find(wrapped), find(~wrapped)];
x = fullAngles(order);
x(x >= 157.5) = x(x >= 157.5) - 180;
colors = [0.06 0.36 0.55; 0.89 0.36 0.13; 0.72 0.55 0.12];
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1380 860]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for channelId = 1:4
    ax = nexttile(layout); hold(ax, 'on');
    for pixelId = 1:3
        plot(ax, x, rates(order,pixelId,channelId), '-o', 'LineWidth', 2, ...
            'MarkerSize', 3.5, 'Color', colors(pixelId,:), ...
            'DisplayName', sprintf('%s (%d,%d)', labels{pixelId}, rows(pixelId), cols(pixelId)));
    end
    grid(ax, 'on'); box(ax, 'off'); xlim(ax, [-22.5 157.5]);
    xticks(ax, [-22.5 0 22.5 45 67.5 90 112.5 135 157.5]);
    xlabel(ax, 'angle (deg; 157.5-180 shifted to -22.5-0)');
    ylabel(ax, 'firing rate (Hz)');
    title(ax, sprintf('%s firing rate', names{channelId}), 'FontWeight', 'bold');
    if channelId == 1, legend(ax, 'Location', 'best', 'Box', 'off'); end
end
title(layout, sprintf('7D NN orientation tuning | model contrast %g, TF 10, SF 2.5', contrastUse), ...
    'FontWeight', 'bold', 'FontSize', 14);
stem = fullfile(outRoot, sprintf('orientation_tuning_contr%d_default3pixels', contrastUse));
saveAll(fig, stem); close(fig);

variableNames = {'angle'};
tableData = fullAngles(:);
for pixelId = 1:3
    for channelId = 1:4
        variableNames{end+1} = sprintf('%s_pixel%d_%d', names{channelId}, rows(pixelId), cols(pixelId)); %#ok<AGROW>
        tableData(:,end+1) = rates(:,pixelId,channelId); %#ok<AGROW>
    end
end
writetable(array2table(tableData, 'VariableNames', variableNames), [stem '.csv']);
save([stem '.mat'], 'fullAngles', 'rates', 'rows', 'cols', 'contrastUse');
end


function plotContrast(outRoot, channels, names, angles, contrasts, angleUse, row0, col0, tag)
[canon, row, col] = mapDefaultPixel(angleUse, row0, col0);
ai = find(abs(angles - canon) < 1e-8, 1);
rates = nan(numel(contrasts), 4);
for channelId = 1:4
    X = channels{channelId};
    rates(:,channelId) = squeeze(X(ai,:,row,col));
end
realC = interp1([0 19 42 66 100], [0 4 10 25 64], contrasts, 'pchip', 'extrap');
colors = [0.06 0.36 0.55; 0.89 0.36 0.13; 0.15 0.55 0.38; 0.72 0.55 0.12];
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1380 860]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for channelId = 1:4
    ax = nexttile(layout);
    plot(ax, realC, rates(:,channelId), '-o', 'LineWidth', 2.2, ...
        'MarkerSize', 6, 'Color', colors(channelId,:), 'MarkerFaceColor', colors(channelId,:));
    grid(ax, 'on'); box(ax, 'off'); set(ax, 'XScale', 'linear', 'YScale', 'linear');
    xlim(ax, [0 64]); xticks(ax, [0 4 10 25 64]);
    if channelId == 4, ylim(ax, [0 35]); yticks(ax, [0 10 20 30 35]); end
    xlabel(ax, 'Real contrast (%)'); ylabel(ax, 'Firing rate (sp/s)');
    title(ax, sprintf('%s firing rate', names{channelId}), 'FontWeight', 'bold');
end
title(layout, sprintf('7D NN contrast tuning | angle %.1f, pixel (%d,%d), TF 10, SF 2.5', ...
    angleUse, row0, col0), 'FontWeight', 'bold', 'FontSize', 14);
stem = fullfile(outRoot, sprintf('contrast_tuning_%s_pixel%d_%d', tag, row0, col0));
saveAll(fig, stem); close(fig);
writetable(array2table([contrasts(:), realC(:), rates], 'VariableNames', ...
    {'model_contrast', 'real_contrast', 'S', 'C', 'I', 'E'}), [stem '.csv']);
save([stem '.mat'], 'contrasts', 'realC', 'rates', 'angleUse', 'row0', 'col0');
end


function [canon, row, col] = mapMiddlePixel(angle)
canon = mod(angle, 45); if canon > 22.5, canon = 45 - canon; end
if angle <= 22.5
    row = 3; col = 9;
elseif angle <= 45
    row = 1; col = 8;
elseif angle <= 67.5
    row = 10; col = 8;
elseif angle <= 90
    row = 3; col = 1;
elseif angle <= 108.75
    row = 8; col = 1;
elseif angle <= 135
    row = 10; col = 3;
elseif angle <= 153.75
    row = 1; col = 3;
elseif abs(angle - 157.5) < 1e-8
    row = 2; col = 3;
else
    row = 8; col = 9;
end
end


function [canon, rowOut, colOut] = mapDefaultPixel(angle, row, col)
if row == 5 && col == 10
    if angle <= 22.5
        canon=angle; rowOut=5; colOut=10;
    elseif angle < 45
        canon=45-angle; rowOut=1; colOut=6;
    elseif angle == 45
        canon=0; rowOut=10; colOut=6;
    elseif angle <= 67.5
        canon=angle-45; rowOut=10; colOut=6;
    elseif angle <= 90
        canon=90-angle; rowOut=5; colOut=1;
    elseif angle < 112.5
        canon=angle-90; rowOut=5; colOut=1;
    elseif angle < 135
        canon=135-angle; rowOut=10; colOut=6;
    elseif angle < 157.5
        canon=angle-135; rowOut=1; colOut=6;
    else
        canon=180-angle; rowOut=5; colOut=10;
    end
    return
end
if row == 1 && col == 10
    if angle <= 22.5
        canon=angle; rowOut=1; colOut=10;
    elseif angle <= 45
        canon=45-angle; rowOut=1; colOut=10;
    elseif angle <= 67.5
        canon=angle-45; rowOut=10; colOut=10;
    elseif angle <= 90
        canon=90-angle; rowOut=1; colOut=1;
    elseif angle <= 112.5
        canon=angle-90; rowOut=10; colOut=1;
    elseif angle <= 135
        canon=135-angle; rowOut=10; colOut=1;
    elseif angle <= 157.5
        canon=angle-135; rowOut=1; colOut=1;
    else
        canon=180-angle; rowOut=10; colOut=10;
    end
    return
end
error('Unsupported default tuning pixel (%d,%d).', row, col);
end


function saveAll(fig, stem)
exportgraphics(fig, [stem '.png'], 'Resolution', 220);
exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector');
savefig(fig, [stem '.fig']);
end
