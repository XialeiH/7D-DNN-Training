function audit_h96_premodel_mapping()
% Compare aggregate/population-specific inputs and raw/transformed h96 outputs.

runtimeDir = getenv('H96_CORRECTED_RUNTIME');
utilsDir = getenv('H96_CURRENT_UTILS');
simulationDir = getenv('SNN_OUTPUT_DIR');
connectionFile = getenv('H96_PIXEL_CONNECTIVITY');
outputRoot = getenv('H96_MAPPING_AUDIT_ROOT');
required = {runtimeDir, utilsDir, simulationDir, connectionFile, outputRoot};
assert(all(~cellfun(@isempty, required)), 'All audit environment variables are required.');
addpath(utilsDir, '-begin');
addpath(runtimeDir, '-begin');
if ~exist(outputRoot, 'dir'); mkdir(outputRoot); end

assertSha256(which('LocalResponse_6D_MLP_prefAngle'), ...
    '41c7fdc837617cd66ee81b0a7f7dc2bc3af45ed8c38e67bc3e85abed8b6bb161');
C = load(connectionFile);

L6Raw = {2.5,3,62.5,86,80.4,106,90,142.2,102,166.7,108,210,118, ...
    {[0 15 29],[0,34.0],[28 32 50],[69 85]}};
L6Pars = {2.5,3,{'c1smooth',L6Raw,unique(0:0.25:120)}};
EKp = {1,0,[50;70;100],[50;60.5;69],'quadratic'};
IKp = struct('Thrsld1',54,'Thrsld2',63,'Highist',103,'Slope',0.941, ...
    'down1',0.3,'IntaH',4.5,'IntaL',-2.5,'IntbL',4,'IntbH',-0.3, ...
    'Mode','multisigmoid','SmoothJoinHalfWidth',[1 0 1], ...
    'SmoothT2QuinticWidth',3);
K = makeL6Kernel(0.75);
wC = 0.3077;
angles = [0,7.5,15,22.5];

networkE = [];
networkI = [];
aggregateRawE = [];
aggregateRawI = [];
aggregateTransformedE = [];
aggregateTransformedI = [];
specificRawE = [];
specificRawI = [];
specificTransformedE = [];
specificTransformedI = [];
angleDeg = [];
normalization = zeros(numel(angles)*2, 6);
row = 0;
for ai = 1:numel(angles)
    for replicate = 1:2
        file = fullfile(simulationDir, sprintf( ...
            'NWSimulationPix_%.1fdeg_rep%02d_h96_validation.mat', ...
            angles(ai), replicate));
        D = load(file, 'NWSmlt');
        N = D.NWSmlt;
        [l4ES,l4EC,l4EI,l4IS,l4IC,l4II,fractions] = ...
            populationInputs(N, C);
        row = row + 1;
        normalization(row,:) = fractions;

        E = N.FS*(1-wC) + N.FC*wC;
        l6 = continuousL6(E, K, L6Pars);
        lgn = squeeze(sum(double(N.PixInptCtgrUse),3));
        lgn = [lgn,zeros(size(lgn,1),1)];
        alpha = angles(ai)*ones(900,1);
        contrast = 100*ones(900,1);

        [aggS,aggC,aggI] = predictAll(alpha,lgn,N.L4E,N.L4E,N.L4E, ...
            N.L4I,N.L4I,N.L4I,l6,contrast);
        [specS,specC,specI] = predictAll(alpha,lgn,l4ES,l4EC,l4EI, ...
            l4IS,l4IC,l4II,l6,contrast);
        aggE = aggS*(1-wC) + aggC*wC;
        specE = specS*(1-wC) + specC*wC;

        networkE = [networkE;E]; %#ok<AGROW>
        networkI = [networkI;N.FI]; %#ok<AGROW>
        aggregateRawE = [aggregateRawE;aggE]; %#ok<AGROW>
        aggregateRawI = [aggregateRawI;aggI]; %#ok<AGROW>
        aggregateTransformedE = [aggregateTransformedE;L6Convert(aggE,EKp)]; %#ok<AGROW>
        aggregateTransformedI = [aggregateTransformedI;InhMulp(aggI,IKp)]; %#ok<AGROW>
        specificRawE = [specificRawE;specE]; %#ok<AGROW>
        specificRawI = [specificRawI;specI]; %#ok<AGROW>
        specificTransformedE = [specificTransformedE;L6Convert(specE,EKp)]; %#ok<AGROW>
        specificTransformedI = [specificTransformedI;InhMulp(specI,IKp)]; %#ok<AGROW>
        angleDeg = [angleDeg;repmat(angles(ai),900,1)]; %#ok<AGROW>
    end
end

variants = struct();
variants.aggregate_raw = summarize(networkE,aggregateRawE,networkI,aggregateRawI);
variants.aggregate_transformed = summarize(networkE,aggregateTransformedE, ...
    networkI,aggregateTransformedI);
variants.specific_raw = summarize(networkE,specificRawE,networkI,specificRawI);
variants.specific_transformed = summarize(networkE,specificTransformedE, ...
    networkI,specificTransformedI);

names = fieldnames(variants);
for i = 1:numel(names)
    fprintf('%s\n', names{i});
    printPopulation('E',variants.(names{i}).E);
    printPopulation('I',variants.(names{i}).I);
end

paired = struct('networkE',networkE,'networkI',networkI, ...
    'aggregateRawE',aggregateRawE,'aggregateRawI',aggregateRawI, ...
    'aggregateTransformedE',aggregateTransformedE, ...
    'aggregateTransformedI',aggregateTransformedI, ...
    'specificRawE',specificRawE,'specificRawI',specificRawI, ...
    'specificTransformedE',specificTransformedE, ...
    'specificTransformedI',specificTransformedI,'angleDeg',angleDeg);
save(fullfile(outputRoot,'h96_mapping_audit.mat'),'variants','paired', ...
    'normalization','-v7.3');

fid = fopen(fullfile(outputRoot,'h96_mapping_audit.txt'),'w');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
for i = 1:numel(names)
    fprintf(fid,'%s\n',names{i});
    writePopulation(fid,'E',variants.(names{i}).E);
    writePopulation(fid,'I',variants.(names{i}).I);
end
end

function [eS,eC,eI,iS,iC,iI,f] = populationInputs(N,C)
se = C.C_SS_Pixel_Us*N.FS + C.C_SC_Pixel_Us*N.FC;
ce = C.C_CS_Pixel_Us*N.FS + C.C_CC_Pixel_Us*N.FC;
ie = C.C_IS_Pixel_Us*N.FS + C.C_IC_Pixel_Us*N.FC;
si = C.C_SI_Pixel_Us*N.FI;
ci = C.C_CI_Pixel_Us*N.FI;
ii = C.C_II_Pixel_Us*N.FI;
eAll = se+ce+ie;
iAll = si+ci+ii;
f = [mean(se./eAll),mean(ce./eAll),mean(ie./eAll), ...
    mean(si./iAll),mean(ci./iAll),mean(ii./iAll)];
eS = se/f(1); eC = ce/f(2); eI = ie/f(3);
iS = si/f(4); iC = ci/f(5); iI = ii/f(6);
end

function l6 = continuousL6(E,K,L6Pars)
field = reshape(E,30,30);
padded = field([end,1:end,1],[end,1:end,1]);
smoothed = conv2(padded,K,'same');
l6 = L6Convert(smoothed(2:end-1,2:end-1),L6Pars)/3;
l6 = min(max(l6(:),1),40);
end

function [S,C,I] = predictAll(alpha,lgn,eS,eC,eI,iS,iC,iI,l6,contrast)
S = predictPopulation('S',alpha,lgn,eS,iS,l6,contrast);
C = predictPopulation('C',alpha,lgn,eC,iC,l6,contrast);
I = predictPopulation('I',alpha,lgn,eI,iI,l6,contrast);
end

function result = predictPopulation(type,alpha,lgn,e,i,l6,contrast)
all = zeros(numel(alpha),5);
for category = 1:5
    all(:,category) = LocalResponse_6D_MLP_prefAngle(type,alpha, ...
        category*ones(size(alpha)),e,i,l6,contrast);
end
result = sum(lgn.*all,2);
end

function result = summarize(nE,mE,nI,mI)
result = struct('E',populationSummary(nE,mE), ...
    'I',populationSummary(nI,mI));
end

function result = populationSummary(network,model)
mask = isfinite(network)&isfinite(model)&network>0;
network=network(mask); model=model(mask);
rel=abs(model-network)./network;
ratio=model./network;
result=struct('count',numel(network),'within33',mean(rel<=0.33), ...
    'within20',mean(rel<=0.20),'ratio0p8to1p25',mean(ratio>=0.8&ratio<=1.25), ...
    'medianRelativeError',median(rel),'meanRelativeError',mean(rel), ...
    'correlation',corr(network,model),'networkMean',mean(network), ...
    'modelMean',mean(model));
end

function printPopulation(label,x)
fprintf([' %s n=%d 33=%.2f%% 20=%.2f%% ratio=%.2f%% med=%.4f ' ...
    'mean=%.4f corr=%.4f rates=%.3f/%.3f\n'],label,x.count,100*x.within33, ...
    100*x.within20,100*x.ratio0p8to1p25,x.medianRelativeError, ...
    x.meanRelativeError,x.correlation,x.networkMean,x.modelMean);
end

function writePopulation(fid,label,x)
fprintf(fid,['%s count=%d within33=%.8f within20=%.8f ratio0p8to1p25=%.8f ' ...
    'medianRelativeError=%.8f meanRelativeError=%.8f correlation=%.8f ' ...
    'networkMean=%.8f modelMean=%.8f\n'],label,x.count,x.within33,x.within20, ...
    x.ratio0p8to1p25,x.medianRelativeError,x.meanRelativeError, ...
    x.correlation,x.networkMean,x.modelMean);
end

function K = makeL6Kernel(sig)
trunc=1.5/sig; n=max(2,floor(2*sig*trunc));
[x,y]=meshgrid(1:n,1:n); center=n/2;
K=exp(-((x-center).^2+(y-center).^2)/(2*sig^2));
K(K<exp(-(trunc^2)/2))=0;
K=K+K(end:-1:1,:)+K(:,end:-1:1)+K(end:-1:1,end:-1:1);
K=K/sum(K,'all'); K=K/((sum(K,'all')-K(2,2))/(1-0.3)); K(2,2)=0.3;
end

function assertSha256(file,expected)
[status,text]=system(sprintf('sha256sum "%s"',file));
assert(status==0&&startsWith(strtrim(text),expected), ...
    'SHA256 mismatch for %s: %s',file,strtrim(text));
end
