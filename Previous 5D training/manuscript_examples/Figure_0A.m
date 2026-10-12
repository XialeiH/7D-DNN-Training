function data = Figure_0A(recompute)
% Contrast-zero solve with the unchanged Figure 1E contrast-100 model.
if nargin < 1; recompute = false; end
out = fileparts(mfilename('fullpath'));
cache = fullfile(out,'Figure_0A_data.mat');
if recompute || ~isfile(cache)
    repo = '/Users/xialeihuang/.codex/repositories/Spectral_Analysis_V1';
    relative = 'project/Spectral_Analysis/matlab-inserting_into_CG_model';
    helpers = fullfile(repo,'Utils',relative,'geometry_analysis/torch_h96_dedupe_array/Utils');
    model = fullfile(repo,'Utils',relative,'l6_next_stage_analysis_20260717');
    runtime = fullfile(repo,'MFpVLDEComput/historical/l6_next_stage_analysis_20260717/real_tuning_l6_i_20260722/mechanism_analysis_20260722/runtime_h96');
    corrected = '/Users/xialeihuang/Desktop/Neuroscience_Project/Spectral_Analysis/matlab-inserting_into_CG_model/Complete_Code_for_Paper3/NYU-Vision-2Drive-main/local_h96_manual_runtime';
    oldPath = path; restore = onCleanup(@()path(oldPath)); %#ok<NASGU>
    addpath(helpers,model,runtime);
    addpath(corrected,'-begin');
    source = fullfile(repo,'Data',relative,'l6_next_stage_analysis_20260717/results_global_bifurcation_20260722/global_bifurcation_setup.mat');
    reference = fullfile(repo,'Data',relative,'report_figure_artifacts/1_Neural_Network_Surrogate/Figure_1F_1/dnn_angle0_contrast100.mat');
    loaded = load(source,'setup');
    original = loaded.setup.Context;
    ref = load(reference,'conditionData');
    assert(all(original.ContrastUse==100) && all(original.OrientationUse==0));
    assert(isequal(original.MapSize,[40 40]));
    assert(max(abs(ref.conditionData.FixedPointMaps.E(:)- ...
        (.6923*original.FixedPoint(1:1600)+.3077*original.FixedPoint(1601:3200))))<1e-10);
    % The continuation archive adds a compensating offset, not part of the
    % original DNN. Remove it for BOTH the reference audit and the new solve.
    original.FixedPointCorrection = [];
    phi100 = l6ns_phi(original.FixedPoint,0,original,[1 1],1);
    audit100 = norm(phi100-original.FixedPoint)/norm(original.FixedPoint);
    fprintf('Uncorrected contrast-100 fixed-point residual: %.12g\n',audit100);
    assert(audit100<1e-5,'Contrast-100 model reconstruction failed.');
    ctx = original;
    ctx.ContrastUse = zeros(size(original.ContrastUse));
    assert(isequal(ctx.PixLGNCtgr,original.PixLGNCtgr));
    n = prod(ctx.MapSize);
    states = zeros(3*n,1001);
    states(:,1) = ctx.InitialState;
    hc = zeros(1001,1); hc(1) = hc_norm(states(:,1));
    updates = nan(1000,1);
    p = ctx.RelaxationP;
    timer = tic;
    for k = 1:1000
        response = l6ns_phi(states(:,k),0,ctx,[1 1],1);
        states(:,k+1) = (1-p)*states(:,k)+p*response;
        assert(all(isfinite(states(:,k+1))),'Nonfinite DNN state.');
        hc(k+1) = hc_norm(states(:,k+1));
        updates(k) = hc_norm(states(:,k+1)-states(:,k))/max(hc(k),eps);
        if mod(k,25)==0
            fprintf('Iteration %d: HC %.8f, relative HC step %.6g\n',k,hc(k+1),updates(k));
        end
        if k>=100 && max(updates(k-2:k))<1e-10; break; end
    end
    iterationSeconds = toc(timer);
    assert(max(updates(k-2:k))<1e-10,'Equilibrium not converged.');
    states = states(:,1:k+1); hc = hc(1:k+1); updates = updates(1:k);
    fixed = states(:,end);
    fixedResidual = hc_norm(l6ns_phi(fixed,0,ctx,[1 1],1)-fixed)/hc_norm(fixed);
    eq = struct('S',fixed(1:n),'C',fixed(n+(1:n)),'I',fixed(2*n+(1:n)));
    fprintf('Computing response Jacobian at contrast zero.\n');
    J = compute_J_h96baseline_pref6D_phi(eq,ctx.ContrastUse,ctx.OrientationUse, ...
        ctx.PixLGNCtgr,ctx.L6Kernel,ctx.L6Parameters, ...
        ctx.C_SS,ctx.C_CS,ctx.C_IS,ctx.C_SC,ctx.C_CC,ctx.C_IC, ...
        ctx.C_SI,ctx.C_CI,ctx.C_II,ctx.L4SEp,ctx.L4SIp,ctx.L4CEp, ...
        ctx.L4CIp,ctx.L4IEp,ctx.L4IIp,4,10,10,p,ctx.EKpUse,ctx.IKpUse,ctx.Isaturation,0);
    rng(20261006);
    direction = randn(3*n,1); direction = direction/norm(direction);
    fdSteps = [1e-2 1e-3 1e-4]; fdErrors = zeros(size(fdSteps));
    for j = 1:numel(fdSteps)
        h = fdSteps(j);
        derivative = (l6ns_phi(fixed+h*direction,0,ctx,[1 1],1)- ...
            l6ns_phi(fixed-h*direction,0,ctx,[1 1],1))/(2*h);
        fdErrors(j) = norm(derivative-J*direction)/max(norm(J*direction),eps);
    end
    fprintf('Directional Jacobian errors: %s\n',mat2str(fdErrors,6));
    assert(min(fdErrors)<1e-4,'Jacobian audit failed.');
    save(fullfile(out,'Figure_0A_equilibrium.mat'),'eq','states','hc','updates','J','ctx','fixedResidual','fdErrors','fdSteps','iterationSeconds','-v7.3');
    fprintf('Computing all 4800 eigenvalues and top cluster.\n');
    [vectors,D] = eig(full(J));
    eigenvalues = diag(D);
    tolerance = ref.conditionData.ClusterTolerance;
    topReal = max(real(eigenvalues));
    selected = real(eigenvalues)>=topReal-tolerance-100*eps(max(abs(topReal),1));
    selectedVectors = vectors(:,selected);
    selectedVectors = selectedVectors./vecnorm(selectedVectors);
    [Q,R] = qr(selectedVectors,0);
    rankTolerance = max(size(R))*eps(max(norm(R,2),1));
    clusterRank = sum(abs(diag(R))>rankTolerance);
    Q = Q(:,1:clusterRank);
    clusterE = .6923*Q(1:n,:)+.3077*Q(n+(1:n),:);
    clusterMap = reshape(sqrt(sum(abs(clusterE).^2,2)),ctx.MapSize);
    eigenResidual = max(vecnorm(J*selectedVectors-selectedVectors.*eigenvalues(selected).'))/max(norm(J,'fro'),eps);
    fprintf('Computing leading singular output mode.\n');
    options = struct('tol',1e-10,'maxit',5000);
    [u,sigma,v,flag] = svds(J,1,'largest',options);
    assert(flag==0);
    [~,pivot] = max(abs(u));
    phase = exp(-1i*angle(u(pivot))); u=u*phase; v=v*phase;
    singularMap = reshape(real(.6923*u(1:n)+.3077*u(n+(1:n))),ctx.MapSize);
    singularResidual = max(norm(J*v-u*sigma),norm(J'*u-v*sigma))/max(sigma,eps);
    assert(eigenResidual<1e-9 && singularResidual<1e-9);
    data = struct('Contrast',0,'Angle',0,'Equilibrium',eq, ...
        'EquilibriumE',reshape(.6923*eq.S+.3077*eq.C,ctx.MapSize), ...
        'Iterations',(0:k)','HCNorm',hc,'RelativeHCUpdates',updates, ...
        'FixedPointResidual',fixedResidual,'Eigenvalues',eigenvalues, ...
        'ClusterTolerance',tolerance,'ClusterCount',nnz(selected), ...
        'ClusterRank',clusterRank,'ClusterMap',clusterMap, ...
        'ClusterEigenvalues',eigenvalues(selected),'TopRealEigenvalue',topReal, ...
        'LeadingSingularValue',sigma,'SingularOutputE',singularMap, ...
        'LeadingLeftSingularVector',u,'LeadingRightSingularVector',v, ...
        'EigenResidual',eigenResidual,'SingularResidual',singularResidual, ...
        'JacobianDirectionalErrors',fdErrors,'JacobianFDSteps',fdSteps, ...
        'ReferenceResidual100',audit100,'ContextSource',source, ...
        'DNNRuntime',which('LocalResponse_6D_MLP_prefAngle'), ...
        'IterationSeconds',iterationSeconds,'RelaxationP',p);
    save(cache,'data','-v7.3');
else
    loaded = load(cache,'data'); data=loaded.data;
end
plot_figure(data,out);
end

function value = hc_norm(state)
n = numel(state)/3;
e = .6923*state(1:n)+.3077*state(n+(1:n));
value = sqrt(mean(.8*e.^2+.2*state(2*n+(1:n)).^2));
end

function plot_figure(d,out)
blue = [33 102 172]/255;
fig = figure('Visible','off','Color','w','Units','inches');
fig.Position = [0 0 36 23];
ax = gobjects(5,1);
positions = [1.6 13 7.6 7.6;13.6 13 7.6 7.6;25.6 13 7.6 7.6;7.6 2 7.6 7.6;20.1 2 7.6 7.6];
for k=1:5
    ax(k)=axes(fig,'Units','inches','Position',positions(k,:), ...
        'FontName','Arial','FontSize',36,'LineWidth',1.5,'Box','on', ...
        'TickDir','out','PositionConstraint','innerposition','NextPlot','add');
end
map_panel(ax(1),d.EquilibriumE,'Equilibrium','sp/s',false);
plot(ax(2),d.Iterations,d.HCNorm,'Color',blue,'LineWidth',3);
xlabel(ax(2),'Iteration'); ylabel(ax(2),'HC norm (sp/s)');
title(ax(2),'Trajectory','FontWeight','bold');
xlim(ax(2),[0 d.Iterations(end)]);
values = d.Eigenvalues(:);
edges = (floor(min(real(values))/.05)-1)*.05:.05:(ceil(max(real(values))/.05)+1)*.05;
counts = histcounts(real(values),edges);
assert(sum(counts)==numel(values));
yyaxis(ax(3),'right');
histogram(ax(3),'BinEdges',edges,'BinCounts',counts,'FaceColor',[.65 .65 .65], ...
    'EdgeColor','none','FaceAlpha',.4);
ylabel(ax(3),'Count'); ax(3).YAxis(2).Color=[.35 .35 .35];
yyaxis(ax(3),'left');
scatter(ax(3),real(values),imag(values),14,blue,'filled');
ylabel(ax(3),'Imaginary part of eigenvalue'); ax(3).YAxis(1).Color='k';
xlabel(ax(3),'Real part of eigenvalue');
title(ax(3),{'Spectrum','histogram map'},'FontWeight','bold');
xline(ax(3),0,':','Color',[.5 .5 .5],'LineWidth',1.5);
xline(ax(3),1,'--','Color',[178 24 43]/255,'LineWidth',2);
xlim(ax(3),[min(edges(1),-.1) max(edges(end),1.05)]);
map_panel(ax(4),d.ClusterMap,'Top eigencluster','',false);
map_panel(ax(5),d.SingularOutputE,'Top singular mode','',true);
set(findall(fig,'-property','FontSize'),'FontSize',36);
for k=1:5
    ax(k).LineWidth=1.5; ax(k).TickDir='out';
    ax(k).Title.FontWeight='bold';
    ax(k).Title.Units='normalized'; ax(k).Title.Position=[.5 1.10 0];
end
drawnow;
% Use physical axes-relative spacing, independent of yyaxis tight insets.
for k=1:5
    ax(k).XLabel.Units='normalized';
    ax(k).XLabel.Position=[.5 -.16 0];
    if k~=3
        ax(k).YLabel.Units='normalized';
        ax(k).YLabel.Position=[-.13 .5 0];
    else
        ax(k).YLabel.Units='normalized';
        ax(k).YLabel.Position=[-.21 .5 0];
    end
end
drawnow;
set(fig,'PaperUnits','inches','PaperSize',[36 23],'PaperPosition',[0 0 36 23]);
print(fig,fullfile(out,'Figure 0A.pdf'),'-dpdf','-painters');
close(fig);
fprintf('Published Figure 0A: HC %.8f, max Re(lambda) %.8f, sigma1 %.8f, cluster n=%d.\n', ...
    d.HCNorm(end),d.TopRealEigenvalue,d.LeadingSingularValue,d.ClusterCount);
end

function map_panel(ax,map,name,units,signed)
imagesc(ax,[0 4],[0 4],map); axis(ax,'image');
set(ax,'YDir','normal','XTick',0:4,'YTick',0:4);
colormap(ax,jet(256));
if signed; clim(ax,max(abs(map(:)))*[-1 1]); end
xlabel(ax,'HC'); ylabel(ax,'HC'); title(ax,name,'FontWeight','bold');
cb=colorbar(ax); cb.Units='inches'; cb.FontSize=36;
pos=ax.Position; ax.Position=pos;
cb.Position=[pos(1)+pos(3)+.5 pos(2) 40/72 pos(4)];
title(cb,units,'FontSize',36,'FontWeight','normal');
end
