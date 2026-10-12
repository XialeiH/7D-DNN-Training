function run_lif_aggregate_condition_tfsf7d_direct_resumable_fixgrid(Angle, Contr, lgnTF, lgnSF, DownFactor, SaveFolder, BlockIndex, SampleCountOverride, DirectRowsOnly)
% Direct condition-to-library generator for h96 TF/SF 7D tests.
% It computes all LGN(1:5) x L6(1:40) responses in one MATLAB process and
% saves the aggregate LDEFrfunc file directly, avoiding 200 repeated raw-LIF startups.
if nargin < 6 || isempty(SaveFolder)
    error('Usage: run_lif_aggregate_condition_tfsf7d_direct_resumable(Angle, Contr, lgnTF, lgnSF, DownFactor, SaveFolder)');
end
if nargin < 7
    BlockIndex = [];
end
if nargin < 8
    SampleCountOverride = [];
end
if nargin < 9 || isempty(DirectRowsOnly)
    DirectRowsOnly = false;
end
DirectRowsOnly = logical(DirectRowsOnly);
if DirectRowsOnly && (isempty(SampleCountOverride) || ~isscalar(SampleCountOverride) || SampleCountOverride < 1)
    error('DirectRowsOnly requires a positive scalar SampleCountOverride.');
end
OneBlockMode = ~isempty(BlockIndex);
if OneBlockMode
    BlockIndex = str2double(string(BlockIndex));
    if ~isfinite(BlockIndex) || BlockIndex < 1 || BlockIndex > 200 || abs(BlockIndex - round(BlockIndex)) > 1e-8
        error('BlockIndex must be an integer in 1..200.');
    end
    BlockIndex = round(BlockIndex);
end
repo = '/scratch/xh2906/NYU-Vision-2Drive-main';
cd(repo);
addpath(repo);
addpath(fullfile(repo, 'Utils'));
addpath(fullfile(repo, 'Data'));
addpath(fullfile(repo, 'MFpVLDEComput'));
addpath(fullfile(repo, 'Data', 'Paper2_NetworkTuning', 'Fig1V4'));
addpath(fullfile(repo, 'Data', 'Paper2_NetworkTuning', 'Fig1V4', 'Paper3LGNSpikeData'));
if ~exist(SaveFolder, 'dir')
    mkdir(SaveFolder);
end
DebugFirstOnly = contains(SaveFolder, 'debug_first');

DataPt = 'V4D2';
ExpTex = 'CtrlL4';
Comment = ['L6Real' ExpTex];
DomPrint = 'Small';
L6Mapctgr = 4;
FlagLargeDom = 1;
L6up = 120;
L6low = 3;
LGNlist = 5;
L6list = 40;
LIFN = 5;
rowsmWin = 6;
colswWin = 3;
realLGN = true;
smth = true;
plotFlag = false;
LGNAngStep = 3.75;
AngAll = 0:LGNAngStep:90;
IGridPolicy = 'original7p5';
I_angles = 90:-7.5:45;
CodePolicy = 'tfsf7d_directrows_Iorig7p5';
lgnTFLabel = tfLabelForFilename(lgnTF);
if Contr == 100
    ContrStr = '';
else
    ContrStr = sprintf('_Contr%d', Contr);
end
AngPrintOut = angleLabelForOutput(Angle);
if DownFactor == 1
    outSuffix = '_Iorig7p5';
else
    outSuffix = sprintf('_LGNstep3p75_DS%d_fixgrid_Iorig7p5', DownFactor);
end
if DirectRowsOnly
    outSuffix = sprintf('%s_directRows%d', outSuffix, round(SampleCountOverride));
end
OutPath = fullfile(SaveFolder, sprintf('Func200%s-%s-Ang%s-SF%.1fTF%s%s-%s%s.mat', DataPt, Comment, AngPrintOut, lgnSF, lgnTFLabel, ContrStr, DomPrint, outSuffix));
if exist(OutPath, 'file')
    fprintf('Aggregate exists, skipping: %s\n', OutPath);
    return;
end
BlockDir = fullfile(SaveFolder, [erase(sprintf('blocks_Ang%s_SF%.1fTF%s%s_DF%d_Iorig7p5', AngPrintOut, lgnSF, lgnTFLabel, ContrStr, DownFactor), filesep)]);
if DirectRowsOnly
    BlockDir = sprintf('%s_directRows%d', BlockDir, round(SampleCountOverride));
end
if ~exist(BlockDir, 'dir')
    mkdir(BlockDir);
end
fprintf('BEGIN resumable direct aggregate: Angle=%.2f Contr=%g TF=%g SF=%g DownFactor=%g -> %s\n', Angle, Contr, lgnTF, lgnSF, DownFactor, OutPath);
fprintf('BlockDir: %s\n', BlockDir);
if OneBlockMode
    fprintf('ONE BLOCK MODE: BlockIndex=%d\n', BlockIndex);
    EarlyLGNctgr = ceil(BlockIndex / L6list);
    EarlyL6ctgr = mod(BlockIndex - 1, L6list) + 1;
    EarlyBlockPath = fullfile(BlockDir, sprintf('block_LGN%02d_L6%02d.mat', EarlyLGNctgr, EarlyL6ctgr));
    if exist(EarlyBlockPath, 'file')
        fprintf('ONE BLOCK MODE early skip existing: %s\n', EarlyBlockPath);
        return;
    end
end

S = load(sprintf('AllMFPixPara_Paper2TuneFig1%s.mat', DataPt));
C_SS_Pixel_Us = S.C_SS_Pixel_Us; C_CS_Pixel_Us = S.C_CS_Pixel_Us; C_IS_Pixel_Us = S.C_IS_Pixel_Us;
C_SC_Pixel_Us = S.C_SC_Pixel_Us; C_CC_Pixel_Us = S.C_CC_Pixel_Us; C_IC_Pixel_Us = S.C_IC_Pixel_Us;
C_SI_Pixel_Us = S.C_SI_Pixel_Us; C_CI_Pixel_Us = S.C_CI_Pixel_Us; C_II_Pixel_Us = S.C_II_Pixel_Us;
NS_L6 = S.NS_L6; NC_L6 = S.NC_L6; NI_L6 = S.NI_L6;
N_HC = S.N_HC; NPixX = S.NPixX; NPixY = S.NPixY;
FrSPixVec = S.FrSPixVec; FrCPixVec = S.FrCPixVec; FrIPixVec = S.FrIPixVec;
S_EE=S.S_EE; S_EI=S.S_EI; S_IE=S.S_IE; S_II=S.S_II; p_EEFail=S.p_EEFail;
S_EL6=S.S_EL6; S_IL6=S.S_IL6;
S_amb=S.S_amb; rS_amb=S.rS_amb; rC_amb=S.rC_amb; rI_amb=S.rI_amb;
S_Elgn=S.S_Elgn; S_Ilgn=S.S_Ilgn;
gL_E=S.gL_E; gL_I=S.gL_I; Ve=S.Ve; Vi=S.Vi; tau_ref=S.tau_ref;
tau_ampa_R=S.tau_ampa_R; tau_ampa_D=S.tau_ampa_D;
tau_nmda_R=S.tau_nmda_R; tau_nmda_D=S.tau_nmda_D;
tau_gaba_R=S.tau_gaba_R; tau_gaba_D=S.tau_gaba_D;
rhoE_ampa=S.rhoE_ampa; rhoE_nmda=S.rhoE_nmda;
rhoI_ampa=S.rhoI_ampa; rhoI_nmda=S.rhoI_nmda;
clear S

Angles_4Input = Angle + [0 45 90 135];
FL6_Angle1 = (L6low:3:L6up) * 1e-3;
NL6S = NS_L6; NL6C = NC_L6; NL6I = NI_L6;

L4SE = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4SI = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4CE = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4CI = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4IE = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4II = zeros(N_HC*NPixX*N_HC*NPixY,1);
for PInd = 1:N_HC*NPixX*N_HC*NPixY
    L4SE(PInd) = (C_SS_Pixel_Us(PInd,:)*FrSPixVec + C_SC_Pixel_Us(PInd,:)*FrCPixVec);
    L4SI(PInd) =  C_SI_Pixel_Us(PInd,:)*FrIPixVec;
    L4CE(PInd) = (C_CS_Pixel_Us(PInd,:)*FrSPixVec + C_CC_Pixel_Us(PInd,:)*FrCPixVec);
    L4CI(PInd) =  C_CI_Pixel_Us(PInd,:)*FrIPixVec;
    L4IE(PInd) = (C_IS_Pixel_Us(PInd,:)*FrSPixVec + C_IC_Pixel_Us(PInd,:)*FrCPixVec);
    L4II(PInd) =  C_II_Pixel_Us(PInd,:)*FrIPixVec;
end
L4Eall = L4SE+L4CE+L4IE;
L4Iall = L4SI+L4CI+L4II;
L4SEp = mean(L4SE./L4Eall); L4CEp = mean(L4CE./L4Eall); L4IEp = mean(L4IE./L4Eall);
L4SIp = mean(L4SI./L4Iall); L4CIp = mean(L4CI./L4Iall); L4IIp = mean(L4II./L4Iall);
LineFit = polyfit(L4Eall,L4Iall,1);
YRange = L4Iall - (LineFit(1)*L4Eall+LineFit(2));
Bdry = floor(max(abs(YRange))/100)*100;
BaseStep = 100;
Step = BaseStep * DownFactor;
Emax = ceil(max(L4Eall)*1.5/Step)*Step;
L4ERange = 0:Step:Emax;
L4IDiffRange = -20*Bdry:Step:20*Bdry;
nE = length(L4ERange);
nI = length(L4IDiffRange);
a0 = nE * nI;
sample_inds = 1:a0;
L4ERcrd = zeros(a0,1);
L4IRcrd = zeros(a0,1);
for LDEInd = 1:a0
    L4EInd = ceil(LDEInd/nI);
    L4IDiffInd = mod(LDEInd,nI);
    if L4IDiffInd == 0
        L4IDiffInd = nI;
    end
    L4EU = L4ERange(L4EInd);
    L4IU = LineFit(1)*L4EU+LineFit(2) + L4IDiffRange(L4IDiffInd);
    L4ERcrd(LDEInd) = L4EU;
    L4IRcrd(LDEInd) = L4IU;
end
L4EPlot = reshape(L4ERcrd, nI, nE);
L4IPlot = reshape(L4IRcrd, nI, nE);
if DirectRowsOnly
    sample_inds = selectSpaceFillingIndices(L4ERcrd, L4IRcrd, round(SampleCountOverride));
end
nSample = numel(sample_inds);
FullStep = BaseStep;
FullEmax = ceil(max(L4Eall)*1.5/FullStep)*FullStep;
FullL4ERange = 0:FullStep:FullEmax;
FullL4IDiffRange = -20*Bdry:FullStep:20*Bdry;
[FullEGrid, FullDGrid] = ndgrid(FullL4ERange, FullL4IDiffRange);
FullIGrid = LineFit(1)*FullEGrid + LineFit(2) + FullDGrid;
OutputGridBounds = [min(FullL4ERange), max(FullL4ERange), min(FullIGrid(:)), max(FullIGrid(:))];
fprintf('Grid DownFactor=%g Step=%g nE=%d nI=%d a0=%d\n', DownFactor, Step, nE, nI, a0);
fprintf('Simulation samples=%d DirectRowsOnly=%d\n', nSample, DirectRowsOnly);
fprintf('Output full-domain grid bounds E=[%.3f %.3f] I=[%.3f %.3f]\n', OutputGridBounds);

NeedParpool = true;
if ~OneBlockMode
    NeedParpool = false;
    for chkLGN = 1:LGNlist
        for chkL6 = 1:L6list
            chkBlockPath = fullfile(BlockDir, sprintf('block_LGN%02d_L6%02d.mat', chkLGN, chkL6));
            if ~exist(chkBlockPath, 'file')
                NeedParpool = true;
                break;
            end
        end
        if NeedParpool
            break;
        end
    end
end
if NeedParpool
    cluster = gcp('nocreate');
    if isempty(cluster)
        jobStorageRoot = getenv('H96_MATLAB_TMP_ROOT');
        if isempty(jobStorageRoot)
            jobStorageRoot = '/scratch/xh2906/h96_tf_sf_7d_dataset_plan_20260702_045654/matlab_tmp';
        end
        jobStorage = fullfile(jobStorageRoot, sprintf('matlab_direct_jobs_%s_%s', getenv('SLURM_JOB_ID'), getenv('SLURM_ARRAY_TASK_ID')));
        if ~exist(jobStorage, 'dir')
            mkdir(jobStorage);
        end
        pc = parcluster('local');
        pc.JobStorageLocation = jobStorage;
        maxWorkers = 128;
        maxWorkersEnv = getenv('H96_PARPOOL_WORKERS');
        if ~isempty(maxWorkersEnv)
            maxWorkers = str2double(maxWorkersEnv);
            if isnan(maxWorkers) || maxWorkers < 4
                error('Invalid H96_PARPOOL_WORKERS=%s', maxWorkersEnv);
            end
        end
        cluster = parpool(pc,[4,maxWorkers]);
    end
    addAttachedFiles(cluster, {'AllMFPixPara_Paper2TuneFig1V4D2.mat'});
else
    fprintf('All block files already exist; assembling without parpool.\n');
end

LDEFrfunc = struct('S',{cell(LGNlist,L6list)}, 'C',{cell(LGNlist,L6list)}, 'I',{cell(LGNlist,L6list)});
DirectSourceBySConfig = containers.Map('KeyType','char','ValueType','char');
LGNSpikeCache = containers.Map('KeyType','char','ValueType','any');
HyperPara = {20*1e3};
if OneBlockMode
    LGNRange = ceil(BlockIndex / L6list);
    L6Range = mod(BlockIndex - 1, L6list) + 1;
else
    LGNRange = 1:LGNlist;
    L6Range = 1:L6list;
end
for LGNctgr = LGNRange
    if LGNctgr == 5
        lgn_S = 0.02;
        lgn_C = 0.02;
        lgn_I = 0.02;
        S_angle = NaN;
        S_configId = NaN;
    else
        lgn_C = cell(2,1);
        lgn_C{1} = loadLGNSpikesCached(LGNSpikeCache, Contr, 1, '0.12', 0, lgnSF, lgnTFLabel);
        lgn_C{2} = loadLGNSpikesCached(LGNSpikeCache, Contr, 2, '0.12', 90, lgnSF, lgnTFLabel);
        S_angle = foldOriTo90(Angles_4Input(LGNctgr));
        if min(abs(AngAll - S_angle)) > 1e-8
            error('S LGN angle %.8f is not on the %.2f-degree grid.', S_angle, LGNAngStep);
        end
        S_configId = round(S_angle / LGNAngStep) + 1;
        lgn_S = cell(2,1);
        lgn_S{1} = loadLGNSpikesCached(LGNSpikeCache, Contr, 4, '0.14', S_angle, lgnSF, lgnTFLabel);
        lgn_S{2} = loadLGNSpikesCached(LGNSpikeCache, Contr, 6, '0.12', S_angle, lgnSF, lgnTFLabel);
        lgn_I = cell(numel(I_angles),1);
        for angId = 1:numel(I_angles)
            lgn_I{angId} = loadLGNSpikesCached(LGNSpikeCache, Contr, 4, '0.14', I_angles(angId), lgnSF, lgnTFLabel);
        end
    end
    fprintf('LGN %d/%d\n', LGNctgr, LGNlist);
    for L6ctgr = L6Range
        tOne = tic;
        BlockPath = fullfile(BlockDir, sprintf('block_LGN%02d_L6%02d.mat', LGNctgr, L6ctgr));
        if exist(BlockPath, 'file')
            if DirectRowsOnly
                if LGNctgr <= 4
                    directKey = sprintf('%d_%d', S_configId, L6ctgr);
                    if ~isKey(DirectSourceBySConfig, directKey)
                        DirectSourceBySConfig(directKey) = BlockPath;
                    end
                end
                fprintf('  L6 %d/%d skipped existing direct rows %.2fs\n', L6ctgr, L6list, toc(tOne));
                if OneBlockMode
                    fprintf('ONE BLOCK MODE complete: existing %s\n', BlockPath);
                    return;
                end
                continue;
            end
            B = load(BlockPath, 'mapS', 'mapC', 'mapI');
            LDEFrfunc.S{LGNctgr,L6ctgr} = B.mapS;
            LDEFrfunc.C{LGNctgr,L6ctgr} = B.mapC;
            LDEFrfunc.I{LGNctgr,L6ctgr} = B.mapI;
            fprintf('  L6 %d/%d skipped existing %.2fs\n', L6ctgr, L6list, toc(tOne));
            if OneBlockMode
                fprintf('ONE BLOCK MODE complete: existing %s\n', BlockPath);
                return;
            end
            continue;
        end
        if DirectRowsOnly && LGNctgr <= 4
            directKey = sprintf('%d_%d', S_configId, L6ctgr);
            if isKey(DirectSourceBySConfig, directKey)
                copyfile(DirectSourceBySConfig(directKey), BlockPath);
                fprintf('  L6 %d/%d reused identical S-angle block %.2fs\n', L6ctgr, L6list, toc(tOne));
                continue;
            end
        end
        StopFlag = (FL6_Angle1(L6ctgr) ~= max(FL6_Angle1, [], 'all'));
        rL6SU = NL6S*FL6_Angle1(L6ctgr);
        rL6CU = NL6C*FL6_Angle1(L6ctgr);
        rL6IU = NL6I*FL6_Angle1(L6ctgr);
        f_sub = cell(nSample,1);
        parfor k = 1:nSample
            LDEInd = sample_inds(k);
            L4EInd = ceil(LDEInd/length(L4IDiffRange));
            L4IDiffInd = mod(LDEInd,length(L4IDiffRange));
            if L4IDiffInd == 0
                L4IDiffInd = length(L4IDiffRange);
            end
            L4EU = L4ERange(L4EInd);
            L4IU = LineFit(1)*L4EU+LineFit(2) + L4IDiffRange(L4IDiffInd);
            if (StopFlag && L4EU > max(L4Eall)*8) || min(L4EU,L4IU)<0
                f_sub{k} = [];
            else
                L4SEU = L4SEp*L4EU; L4CEU = L4CEp*L4EU; L4IEU = L4IEp*L4EU;
                L4SIU = L4SIp*L4IU; L4CIU = L4CIp*L4IU; L4IIU = L4IIp*L4IU;
                f_sub{k} = LIFo_SinglePixel_RealLGN(lgnTF, L4SEU,L4SIU, L4CEU,L4CIU, L4IEU,L4IIU, S_EE,S_EI,S_IE,S_II,p_EEFail, S_EL6,S_IL6,rL6SU,rL6CU,rL6IU,S_amb,rS_amb,rC_amb,rI_amb, lgn_S,lgn_C,lgn_I, S_Elgn,S_Ilgn, gL_E,gL_I,Ve,Vi,tau_ref, tau_ampa_R,tau_ampa_D,tau_nmda_R,tau_nmda_D,tau_gaba_R,tau_gaba_D, rhoE_ampa,rhoE_nmda,rhoI_ampa,rhoI_nmda, HyperPara);
            end
        end
        FrSample = nan(nSample, LIFN);
        fSubSize = nan(nSample, 2);
        fSubNonempty = false(nSample, 1);
        for sampleId = 1:nSample
            if isempty(f_sub{sampleId})
                continue;
            end
            fSubNonempty(sampleId) = true;
            fSubSize(sampleId,:) = size(f_sub{sampleId});
            if size(f_sub{sampleId},1) ~= LIFN
                continue;
            end
            nT = size(f_sub{sampleId},2);
            cStart = max(1, floor(nT/2));
            FrSample(sampleId,:) = (mean(f_sub{sampleId}(:,cStart:nT),2))';
        end
        FrSample(FrSample<0) = eps;
        if DirectRowsOnly
            directL4E = L4ERcrd(sample_inds);
            directL4I = L4IRcrd(sample_inds);
            directS = 0.75*FrSample(:,1) + 0.25*FrSample(:,2);
            directC = 0.5*FrSample(:,3) + 0.5*FrSample(:,4);
            directI = FrSample(:,5);
            LGNLoadedFileCount = LGNSpikeCache.Count;
            TempBlockPath = [tempname(BlockDir) '.mat'];
            save(TempBlockPath, 'directL4E', 'directL4I', 'directS', 'directC', 'directI', ...
                'sample_inds', 'SampleCountOverride', 'LGNctgr', 'L6ctgr', 'Angle', ...
                'Contr', 'lgnTF', 'lgnSF', 'DownFactor', 'LGNAngStep', 'S_angle', ...
                'I_angles', 'IGridPolicy', 'CodePolicy', 'LGNLoadedFileCount', '-v7.3');
            [MoveOK, MoveMessage] = movefile(TempBlockPath, BlockPath, 'f');
            if ~MoveOK
                error('Could not finalize direct-row block %s: %s', BlockPath, MoveMessage);
            end
            if LGNctgr <= 4
                DirectSourceBySConfig(sprintf('%d_%d', S_configId, L6ctgr)) = BlockPath;
            end
            if OneBlockMode
                fprintf('ONE BLOCK MODE saved direct rows: %s %.2fs\n', BlockPath, toc(tOne));
                return;
            end
            fprintf('  L6 %d/%d direct rows done %.2fs\n', L6ctgr, L6list, toc(tOne));
            continue;
        end
        FrLDE = nan(a0, LIFN);
        FrLDE(sample_inds,:) = FrSample;
        if DownFactor > 1
            sparseSmth = false;
        else
            sparseSmth = smth;
        end
        [vecS, ~] = LDEPlotRateFunc_RealLGN_NoPlot(L4EPlot,L4IPlot,FrLDE,nE,nI,LGNctgr,L6ctgr,'S',sparseSmth,plotFlag,realLGN,[rowsmWin, colswWin]);
        [vecC, ~] = LDEPlotRateFunc_RealLGN_NoPlot(L4EPlot,L4IPlot,FrLDE,nE,nI,LGNctgr,L6ctgr,'C',sparseSmth,plotFlag,realLGN,[rowsmWin, colswWin]);
        [vecI, ~] = LDEPlotRateFunc_RealLGN_NoPlot(L4EPlot,L4IPlot,FrLDE,nE,nI,LGNctgr,L6ctgr,'I',sparseSmth,plotFlag,realLGN,[rowsmWin, colswWin]);
        mapS = gridFromVec(vecS, L4ERcrd, L4IRcrd, OutputGridBounds);
        mapC = gridFromVec(vecC, L4ERcrd, L4IRcrd, OutputGridBounds);
        mapI = gridFromVec(vecI, L4ERcrd, L4IRcrd, OutputGridBounds);
        if DownFactor > 1 && smth
            mapS = smoothDenseMapOmitNan(mapS, rowsmWin, colswWin);
            mapC = smoothDenseMapOmitNan(mapC, rowsmWin, colswWin);
            mapI = smoothDenseMapOmitNan(mapI, rowsmWin, colswWin);
        end
        LDEFrfunc.S{LGNctgr,L6ctgr} = mapS;
        LDEFrfunc.C{LGNctgr,L6ctgr} = mapC;
        LDEFrfunc.I{LGNctgr,L6ctgr} = mapI;
        TempBlockPath = [tempname(BlockDir) '.mat'];
        save(TempBlockPath, 'mapS', 'mapC', 'mapI', 'LGNctgr', 'L6ctgr', ...
            'Angle', 'Contr', 'lgnTF', 'lgnSF', 'DownFactor', 'LGNAngStep', ...
            'I_angles', 'IGridPolicy', 'CodePolicy', '-v7.3');
        [MoveOK, MoveMessage] = movefile(TempBlockPath, BlockPath, 'f');
        if ~MoveOK
            error('Could not finalize full-library block %s: %s', BlockPath, MoveMessage);
        end
        if OneBlockMode
            fprintf('ONE BLOCK MODE saved: %s %.2fs\n', BlockPath, toc(tOne));
            return;
        end
        if DebugFirstOnly && LGNctgr == 1 && L6ctgr == 1
            debugPath = fullfile(SaveFolder, 'debug_first_LGN1_L6_1.mat');
            save(debugPath, 'fSubSize', 'fSubNonempty', 'FrLDE', 'vecS', 'vecC', 'vecI', 'mapS', 'mapC', 'mapI', 'L4ERcrd', 'L4IRcrd', 'L4EPlot', 'L4IPlot', 'DownFactor', '-v7.3');
            fprintf('DEBUG first-only saved %s finite FrLDE=%d vecS=%d mapS=%d\n', debugPath, nnz(isfinite(FrLDE)), nnz(isfinite(vecS)), nnz(isfinite(mapS)));
            return;
        end
        fprintf('  L6 %d/%d done %.2fs\n', L6ctgr, L6list, toc(tOne));
    end
end
if DirectRowsOnly
    save(OutPath, 'Angle', 'Contr', 'lgnTF', 'lgnSF', 'DownFactor', ...
        'SampleCountOverride', 'sample_inds', 'LGNAngStep', 'I_angles', ...
        'IGridPolicy', 'CodePolicy', '-v7.3');
    fprintf('Saved direct-row completion marker %s\n', OutPath);
    return;
end
if abs(Angle - 22.5) < 1e-8
    LDEFrfuncUse.S = FuncSwap(LDEFrfunc.S,[1,2,3,4;4,3,2,1]);
    LDEFrfuncUse.C = FuncSwap(LDEFrfunc.C,[1,2,3,4;4,3,2,1]);
    LDEFrfuncUse.I = FuncSwap(LDEFrfunc.I,[1,2,3,4;4,3,2,1]);
    LDEFrfunc = LDEFrfuncUse;
end
missingBlocks = {};
for chkLGN = 1:LGNlist
    for chkL6 = 1:L6list
        if isempty(LDEFrfunc.S{chkLGN,chkL6}) || isempty(LDEFrfunc.C{chkLGN,chkL6}) || isempty(LDEFrfunc.I{chkLGN,chkL6})
            missingBlocks{end+1} = sprintf('LGN%02d_L6%02d', chkLGN, chkL6); %#ok<AGROW>
        end
    end
end
if ~isempty(missingBlocks)
    error('Cannot assemble final aggregate; missing %d blocks, first=%s', numel(missingBlocks), missingBlocks{1});
end
L4EfuncX = linspace(OutputGridBounds(1), OutputGridBounds(2), 400);
L4IfuncY = linspace(OutputGridBounds(3), OutputGridBounds(4), 300);
[L4EmeshX, L4ImeshY] = meshgrid(L4EfuncX, L4IfuncY);
save(OutPath, 'LDEFrfunc', 'L4EmeshX', 'L4ImeshY', 'lgnTF', 'lgnSF', 'DownFactor', 'OutputGridBounds', '-v7.3');
fprintf('Saved %s\n', OutPath);
end

function M = gridFromVec(vec, x, y, outputBounds)
valid = isfinite(vec) & isfinite(x) & isfinite(y);
L4EfuncX = linspace(outputBounds(1), outputBounds(2), 400);
L4IfuncY = linspace(outputBounds(3), outputBounds(4), 300);
[X, Y] = meshgrid(L4EfuncX, L4IfuncY);
if nnz(valid) < 3
    M = nan(size(X));
else
    M = griddata(x(valid), y(valid), vec(valid), X, Y, 'Linear');
end
end

function label = tfLabelForFilename(tf)
if abs(tf - round(tf)) < 1e-8
    label = sprintf('%d', round(tf));
else
    label = regexprep(regexprep(sprintf('%.4f', tf), '0+$', ''), '\.$', '');
end
end

function label = angleLabelForOutput(angleValue)
if abs(angleValue*2 - round(angleValue*2)) < 1e-8
    label = sprintf('%.1f', angleValue);
else
    label = regexprep(regexprep(sprintf('%.2f', angleValue), '0+$', ''), '\.$', '');
end
end

function fmt = angleFmtForLGNFile(theta)
% Preserve original LGN filename convention for old 7.5-degree grid angles.
if abs(theta / 7.5 - round(theta / 7.5)) < 1e-8
    fmt = '%.1f';
else
    fmt = '%.2f';
end
end

function theta = foldOriTo90(theta)
theta = mod(theta, 180);
if theta > 90
    theta = 180 - theta;
end
end

function spikes = loadLGNSpikesCached(cache, contr, cellN, rowText, theta, lgnSF, lgnTFLabel)
key = sprintf('C%d_Cell%d_A%.8f_SF%.8f_TF%s', contr, cellN, theta, lgnSF, lgnTFLabel);
if isKey(cache, key)
    spikes = cache(key);
    return;
end
angleFmt = angleFmtForLGNFile(theta);
if contr == 100
    fileName = sprintf(['LGN_Cell%d_Row%s_Col0.15_deg' angleFmt '_SF%.1f_TF%s_TSimu25.mat'], ...
        cellN, rowText, theta, lgnSF, lgnTFLabel);
else
    fileName = sprintf(['LGN_Contr%d_Cell%d_Row%s_Col0.15_deg' angleFmt '_SF%.1f_TF%s_TSimu25.mat'], ...
        contr, cellN, rowText, theta, lgnSF, lgnTFLabel);
end
if exist(fileName, 'file') ~= 2
    error('Missing LGN spike file: %s', fileName);
end
data = load(fileName, 'SpTimeMix');
spikes = sort(data.SpTimeMix);
cache(key) = spikes;
end

function M = smoothDenseMapOmitNan(M, rowWinSize, colWinSize)
% Sparse direct grids can contain NaNs outside the interpolation hull; avoid
% propagating those NaNs across the whole dense response surface.
M = smoothdata(M, 2, 'movmean', rowWinSize, 'omitnan');
M = smoothdata(M, 1, 'movmean', colWinSize, 'omitnan');
end

function sampleInds = selectSpaceFillingIndices(x, y, sampleCount)
valid = find(isfinite(x) & isfinite(y) & x >= 0 & y >= 0);
if isempty(valid)
    error('No valid L4E/L4I coordinates available for direct-row sampling.');
end
sampleCount = min(sampleCount, numel(valid));
xv = x(valid);
yv = y(valid);
xn = (xv - min(xv)) ./ max(max(xv) - min(xv), eps);
yn = (yv - min(yv)) ./ max(max(yv) - min(yv), eps);
selected = false(numel(valid), 1);
[~, first] = min(xn.^2 + yn.^2);
selected(first) = true;
minDist = (xn - xn(first)).^2 + (yn - yn(first)).^2;
for k = 2:sampleCount
    minDist(selected) = -inf;
    [~, next] = max(minDist);
    selected(next) = true;
    newDist = (xn - xn(next)).^2 + (yn - yn(next)).^2;
    minDist = min(minDist, newDist);
end
sampleInds = sort(valid(selected));
end
