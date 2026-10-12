function plot_1_2_premodel_vs_nn_cg()
% Plot real Paper2-SNN versus corrected-h96 paired firing-rate results.

dataFile = getenv('PREMODEL_PAIR_DATA');
outputDir = getenv('FIGURE_PDF_DIR');
artifactDir = getenv('FIGURE_ARTIFACT_DIR');
if isempty(dataFile) || isempty(outputDir) || isempty(artifactDir)
    error('PREMODEL_PAIR_DATA, FIGURE_PDF_DIR, and FIGURE_ARTIFACT_DIR are required.');
end
assert(isfile(dataFile), 'Missing paired data: %s', dataFile);
if ~exist(outputDir,'dir'); mkdir(outputDir); end
if ~exist(artifactDir,'dir'); mkdir(artifactDir); end
D = load(dataFile,'paired','provenance');
P = D.paired;

factors = [4/3, 1.2];
figureNames = {'1.2_SNN_premodel_vs_NN_CG_pixelwise_firing_rates', ...
    '1.3_Premodel_vs_NN_CG_pixelwise_firing_rates_20pct_error'};
labels = {'33%', '20%'};
statsFile = fullfile(artifactDir,'1.2_1.3_error_band_proportions.txt');
fid = fopen(statsFile,'w'); cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'Data: %s\nPoints per population: %d (3600 base + 1080 real 30%% extension)\n\n', ...
    dataFile,numel(P.E_premodel));

for k = 1:numel(factors)
    factor = factors(k);
    eInside = P.E_h96 >= P.E_premodel/factor & P.E_h96 <= factor*P.E_premodel;
    iInside = P.I_h96 >= P.I_premodel/factor & P.I_h96 <= factor*P.I_premodel;
    fprintf('%s: E %d/%d = %.6f%%; I %d/%d = %.6f%%\n', labels{k}, ...
        sum(eInside),numel(eInside),100*mean(eInside), ...
        sum(iInside),numel(iInside),100*mean(iInside));
    fprintf(fid,'%s multiplicative band [x/%.10g, %.10g*x]\n',labels{k},factor,factor);
    fprintf(fid,'E\t%d\t%d\t%.6f%%\n',sum(eInside),numel(eInside),100*mean(eInside));
    fprintf(fid,'I\t%d\t%d\t%.6f%%\n\n',sum(iInside),numel(iInside),100*mean(iInside));
    generateFigure(P,factor,figureNames{k},outputDir,artifactDir);
end

copyfile(dataFile,fullfile(artifactDir,'1.2_1.3_real_paired_data.mat'));
save(fullfile(artifactDir,'1.2_1.3_plot_provenance.mat'),'D','-v7.3');
end

function generateFigure(P,errorFactor,figureName,outputDir,artifactDir)
fig = figure('Visible','off','Color','w','Units','inches','Position',[1 1 13.2 6.1]);
t = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
axE = nexttile(t);
drawPanel(axE,P.E_premodel,P.E_h96,[5 40],'E',[1 0.125 0.125],errorFactor);
xticks(axE,[10 20 30 40]); yticks(axE,5:5:40);
axI = nexttile(t);
drawPanel(axI,P.I_premodel,P.I_h96,[25 110],'I',[0.21 0.36 1],errorFactor);
xticks(axI,[40 60 80 100]); yticks(axI,30:10:110);

pdfFile = fullfile(outputDir,[figureName '.pdf']);
artifactBase = fullfile(artifactDir,figureName);
savefig(fig,[artifactBase '.fig']);
exportgraphics(fig,[artifactBase '.png'],'Resolution',600);
exportgraphics(fig,pdfFile,'ContentType','vector');
close(fig);
fprintf('Saved %s\n',pdfFile);
end

function drawPanel(ax,xData,yData,limits,population,pointColor,errorFactor)
hold(ax,'on');
lineX = linspace(limits(1),limits(2),500);
plot(ax,lineX,lineX,'k--','LineWidth',1.1);
plot(ax,lineX,errorFactor*lineX,'k--','LineWidth',1.5);
plot(ax,lineX,lineX/errorFactor,'k--','LineWidth',1.5);
scatter(ax,xData,yData,1,'o','filled','MarkerFaceColor',pointColor, ...
    'MarkerEdgeColor','none','MarkerFaceAlpha',1);
xlim(ax,limits); ylim(ax,limits); axis(ax,'square');
title(ax,sprintf('Pixelwise %s firing rates',population),'FontSize',20,'FontWeight','bold');
xlabel(ax,'Premodel Fr (Hz)','FontSize',18);
if population == 'E'; ylabel(ax,'NN-CG Model Fr (Hz)','FontSize',18); end
set(ax,'FontName','Arial','FontSize',15,'LineWidth',1.2,'Box','off', ...
    'TickDir','in','Layer','top');
disableDefaultInteractivity(ax);
toolbarHandle = axtoolbar(ax,{}); toolbarHandle.Visible = 'off';
end
