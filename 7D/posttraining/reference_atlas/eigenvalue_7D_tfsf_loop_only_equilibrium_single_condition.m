%% 6D pref-angle equilibrium comparison
% Local comparison driver for the bs8192 pref-angle ReLU checkpoints.
% By default it runs the exact library-vs-NN equilibrium compare for:
%   angle = 7.50 deg
%   contrast = 80
%
% Geometry, connectivity, IC policy, and library lookup stay aligned with
% the existing equilibrium script. Only the NN response operator is
% replaced with the new 6D pref-angle adapter stack in Utils/.

%% block 1
tic

CurrentFolder = '/scratch/xh2906/NYU-Vision-2Drive-main';
addpath(CurrentFolder)
addpath([CurrentFolder '/Utils'])
addpath([CurrentFolder '/Data'])
if exist('RuntimeBundleDirOverride', 'var') && ~isempty(RuntimeBundleDirOverride)
    addpath(RuntimeBundleDirOverride, '-begin');
end
SaveToFolder = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/Paper3PlotingData/']; % V1D2
if exist('SaveToFolderOverride', 'var') && ~isempty(SaveToFolderOverride)
    SaveToFolder = SaveToFolderOverride;
end
DataFolder = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/25Function_Binocular/']; % V1D2
DataFolder1 = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/16Function_Scheme/LARGE16/']; % V1D2
DataFolder2 = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/25Function_binocular_realLGN/']; % V1D2
DataFolder3 = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/Paper2PlotingData/Typical_trajs/']; % V1D2
DataFolder4 = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/Paper3ICTestData/'];

addpath('/Users/xialeihuang/Desktop/Neuroscience_Project/Spectral_Analysis/matlab-inserting_into_CG_model/Complete_Code_for_Paper3/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/Paper3PlotingData/Angle grid 3.75 computed locally');
% addpath('/Users/xialeihuang/Desktop/Neuroscience_Project/Spectral_Analysis/matlab-inserting_into_CG_model/Complete_Code_for_Paper3/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/Paper3PlotingData/Angle grid 3.75 computed HPC');
addpath([CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/'])

addpath(DataFolder4)
addpath(DataFolder3)
addpath(SaveToFolder)
addpath(DataFolder2)
addpath(DataFolder1)
addpath(DataFolder)
addpath([CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/'])

%% block 2
DataPt = 'V4D2'; %'V4D1'; 'V5D1'
load(sprintf("AllMFPixPara_Paper2TuneFig1%s.mat",DataPt))

CurrentFolder = '/scratch/xh2906/NYU-Vision-2Drive-main';
FigurePaperPath = [CurrentFolder '/Figures/Demo062324/']; % V1D2
MonocuFlag = false;
close all

%% block 3
close all
xEAll = 0:0.2:1; xIAll = 0:0.2:1;
if ~exist('lgnSFOverride', 'var') || isempty(lgnSFOverride)
    lgnSF = 2.5;
else
    lgnSF = lgnSFOverride;
end
if ~exist('lgnTFOverride', 'var') || isempty(lgnTFOverride)
    lgnTF = 10;
else
    lgnTF = lgnTFOverride;
end
if abs(lgnTF - round(lgnTF)) < 1e-8
    lgnTFLabel = sprintf('%d', round(lgnTF));
else
    lgnTFLabel = regexprep(regexprep(sprintf('%.4f', lgnTF), '0+$', ''), '\.$', '');
end
global H96_LGN_TF_USE H96_LGN_SF_USE
H96_LGN_TF_USE = lgnTF;
H96_LGN_SF_USE = lgnSF;

ExpTex = 'CtrlL4';
Comment = ['L6Real' ExpTex];

L6pars = {};
L6pars{3} = {2.5,3,[28; 40; 50],[62.5; 84; 96],'quadratic'};
L6pars{1} = {2.5,3,62.5,86,86};
L6pars{2} = {2.5,3,62.5,86,80.4,106,90,142.2, 102, 166.7, 108, 210, 118};

L6ParamRawUse = {2.5,3,62.5,86,80.4,106,90,142.2,102,166.7,108,210,118,{[0 15 29],[0,34.0],[28 32 50],[69 85]}};
L6C1Grid = 0:0.25:120;
L6ParamUse = {2.5,3,{'c1smooth',L6ParamRawUse,L6C1Grid}};
L6pars{5} = L6ParamUse;
 % 142.2; 166.7

% figure(101)
% xx = 0:0.1:100;
% hold on
% for L6Id = 1:3
%     yy = L6Convert(xx,L6pars{L6Id});
%     plot(xx,yy,'DisplayName',sprintf('L6Convert-%d',L6Id))
% end
% grid on
% xlim([0 90])
% ylim([0 120])
% xlabel('L4E')
% ylabel('frL6')
% legend('Location','SouthEast')

IKp3.Thrsld1 = 54; IKp3.Thrsld2 = 63;
IKp3.Highist = 103;  IKp3.Slope = 0.941; % this is for new rule: multiplicative saturation
IKp3.down1 = 0.3; % how much does the first part goes down -- a number between 0 and 1
Isaturation = true;
IKp3.Mode = 'multisigmoid';
IKp3.SmoothJoinHalfWidth = [1 0 1];
IKp3.SmoothT2QuinticWidth = 3; % or 'linear'
IKp3.IntaH = 4.5; IKp3.IntaL = -2.5;
IKp3.IntbL = 4; IKp3.IntbH = -0.3; % one side start from 3. another side start from 6

IKpUse = IKp3;

%% block 4
close all
% figure(102)
EKpUse = {1,0,[50; 70; 100],[50; 60.5; 69],'quadratic'}; 
% % we are using the L4eL6 conversion curve formula for L4E depression
% xx = 0:0.1:220;
% hold on
%     yy = L6Convert(xx,EKpUse);
%     plot(xx,yy,'DisplayName',sprintf('L4E-depression'))
%     xlabel('L4E')
% ylabel('L4E-dep')
% grid on

%% block 5
close all
%Comment = '';
xEInd = 6%:length(xEAll)
xIInd = 6
fprintf('xE=%.1f, xI=%.1f\n',xEAll(xEInd),xIAll(xIInd))
L6parId = 5;fprintf('L6 ParameterID %d\n',L6parId)
if iscell(L6pars{L6parId}{end}) && numel(L6pars{L6parId}{end}) >= 1 && strcmp(L6pars{L6parId}{end}{1}, 'weightedblend')
    L6Comment = sprintf('L6wbR%.2f', L6pars{L6parId}{end}{2})
elseif ischar(L6pars{L6parId}{end}) && strcmp(L6pars{L6parId}{end}, 'quadratic')
    L6Comment = sprintf('%s',L6pars{L6parId}{end})
elseif iscell(L6pars{L6parId}{end})
    L6Comment = sprintf('LowHighQuadr5')
else
    L6Comment = sprintf('linear')
end
tic

% Default single-case compare for the new 6D pref-angle network.
if ~exist('AngleListOverride', 'var') || isempty(AngleListOverride)
    AngleList = {'7.50'};
else
    AngleList = AngleListOverride;
end
DomList = {'Small'};

% Auto-discover contrasts from existing precomputed files (fallback to known set).
funcScan = dir(fullfile(SaveToFolder, sprintf('Func200%s-%s-Ang*-SF%.1fTF%s_Contr*-Small_part.mat', ...
    DataPt, Comment, lgnSF, lgnTFLabel)));
contrFromFiles = [];
for fId = 1:numel(funcScan)
    tok = regexp(funcScan(fId).name, '_Contr(\d+)-', 'tokens', 'once');
    if ~isempty(tok)
        contrFromFiles(end+1) = str2double(tok{1}); %#ok<AGROW>
    end
end
if isempty(contrFromFiles)
    ContrListAll = [5 14 24 33 42 52 61 71 80 89 95];
else
    ContrListAll = unique(sort(contrFromFiles));
end
if ~exist('ContrListOverride', 'var') || isempty(ContrListOverride)
    RequestedContrList = 80;
else
    RequestedContrList = ContrListOverride;
end
if ~all(ismember(RequestedContrList, ContrListAll))
    error('One or more requested contrasts were not found in the library file set.');
end
ContrList = RequestedContrList;
fprintf('Using contrasts: %s\n', mat2str(ContrList));

%% block 6
% there are 3200 pixels in total
N_HCOutX = 4; N_HCOutY = 4; PixNumOut = N_HCOutX*N_HCOutY*NPixX*NPixY;
N_HCinX = 4; N_HCinY = 4;
NPixX = 10; NPixY = 10;

%% block 7
% here I test multiple angles,
% For 5D MLP, orientation is an explicit input dimension.
AngleTestAll = 0:3.75:22.5;
TestNum = length(AngleTestAll); ICTestAll = cell(TestNum,1);
TestExSeq = xEAll(xEInd)*ones(size(0:3.75:180)); TestEySeq = 1*ones(size(TestExSeq));
TestIxSeq = xIAll(xIInd)*ones(size(0:3.75:180)); TestIySeq = 1*ones(size(TestExSeq));

%% block 8
for k = 1:length(AngleList)
    angle = str2double(AngleList{k});

    % Fixed IC policy: use the same 0.0-degree equilibrium seed for all runs.
    filename = fullfile(CurrentFolder, 'Data', 'Paper2_NetworkTuning', 'Fig1V4', ...
        'LDETracesV4D2_NewSmear_ang0.0.mat');
    if ~isfile(filename)
        error('Required IC file not found: %s', filename);
    end
    fprintf('Angle %.2f using IC file: %s\n', angle, filename);

    ICData = load(filename);
    LDEICPart = ICData.LDEEquv;
LDEICOut = HCRotXY(LDEICPart,0,N_HCinX,N_HCinY,N_HCOutX,N_HCOutY,NPixX,NPixY,0);
for TestId = 1:TestNum
    ICTestAll{TestId} = LDEICOut;
end
LDEOutAll = cell(size(ICTestAll));
NANFlag = false(size(ICTestAll));
%BGFlagAllTest = cell(size(ICTestAll));

%% block 9
%PlotUse = load([DataFolder2 sprintf('LDETraces_bg-ang%s.mat',AngPrint)],'LDEEquv');
%LDEEquv = PlotUse.LDEEquv;
%LDEEquv = HCRot(LDEEquv,RotInd,N_HCOut,NPixX,NPixY,MirInd);

%% block 10
p = 0.33;% 1-p for the original input
EpocTest = 300;
ExportFlag = 'xn';

L2Diff_OneStepAll = zeros(length(ICTestAll),EpocTest+1);
DiffVecAll = zeros(length(ICTestAll),EpocTest+1,3*NPixY*NPixX);

%% block 11
% Determine L4 Input from a range
L4SE = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4SI = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4CE = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4CI = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4IE = zeros(N_HC*NPixX*N_HC*NPixY,1);
L4II = zeros(N_HC*NPixX*N_HC*NPixY,1);
for PInd = 1:N_HC*NPixX*N_HC*NPixY
    L4SE(PInd) = (C_SS_Pixel_Us(PInd,:)*FrSPixVec          + C_SC_Pixel_Us(PInd,:)*FrCPixVec);
    L4SI(PInd) =  C_SI_Pixel_Us(PInd,:)*FrIPixVec ;
    L4CE(PInd) = (C_CS_Pixel_Us(PInd,:)*FrSPixVec          + C_CC_Pixel_Us(PInd,:)*FrCPixVec);
    L4CI(PInd) =  C_CI_Pixel_Us(PInd,:)*FrIPixVec ;
    L4IE(PInd) = (C_IS_Pixel_Us(PInd,:)*FrSPixVec          + C_IC_Pixel_Us(PInd,:)*FrCPixVec);
    L4II(PInd) =  C_II_Pixel_Us(PInd,:)*FrIPixVec ;
end
L4Eall = L4SE+L4CE+L4IE; L4Iall = L4SI+L4CI+L4II;
L4SEp = mean(L4SE./L4Eall);L4CEp = mean(L4CE./L4Eall);L4IEp = mean(L4IE./L4Eall);
L4SIp = mean(L4SI./L4Iall);L4CIp = mean(L4CI./L4Iall);L4IIp = mean(L4II./L4Iall);

%% block 12
for Contr = ContrList
    if Contr == 100
        ContrStr = '';
    else
        ContrStr = sprintf('_Contr%d',Contr);
    end
    fprintf('Running contrast %d\n', Contr);

    EpocTest = 300;
    LDEOutAll = cell(size(ICTestAll));
    LDEOut = cell(size(ICTestAll));
    NANFlag = false(size(ICTestAll));
    L2Diff_OneStepAll = zeros(length(ICTestAll),EpocTest+1);
    DiffVecAll = zeros(length(ICTestAll),EpocTest+1,3*NPixY*NPixX);
    err_ode_true_all = zeros(length(ICTestAll),EpocTest+1);

AngleInpt = angle;

RotInd = floor(AngleInpt/45); % 0-3
MirInd = mod(floor(AngleInpt/22.5),2); % 0: no mirror; 1: mirror
switch MirInd % find the corresponding source
    case 0
        AngSource = mod(AngleInpt,45);
    case 1
        AngSource = mod(-AngleInpt,45);
end

% Build candidate function-angle list from existing precomputed files.
funcAnglePattern = sprintf('Func200%s-%s-Ang*-SF%.1fTF%s%s-%s*.mat', ...
    DataPt, Comment, lgnSF, lgnTFLabel, ContrStr, DomList{1});
funcAngleFiles = dir(fullfile(SaveToFolder, funcAnglePattern));
AvailAng = [];
for fId = 1:numel(funcAngleFiles)
    tok = regexp(funcAngleFiles(fId).name, '-Ang([0-9]+(?:\.[0-9]+)?)-SF', 'tokens', 'once');
    if ~isempty(tok)
        AvailAng(end+1) = str2double(tok{1}); %#ok<AGROW>
    end
end
if isempty(AvailAng)
    error('No precomputed function files found for pattern: %s', funcAnglePattern);
end
AvailAng = unique(sort(AvailAng));

[~,AngFuncCtgr] = min(abs(AvailAng - AngSource));
Angle = AvailAng(AngFuncCtgr);
AngPrint = sprintf('%.2f', Angle);
fprintf('Input angle %.2f -> source %.2f -> using precomputed angle %.2f\n', ...
    AngleInpt, AngSource, Angle);

DefaultFull70LibCacheDir = '/scratch/xh2906/librarySCI_runs/library_only_70_fullnods_oldform_mlxorder_20260511_104242/fullnods_smooth2d_contr14_24_20260514_025512';
DefaultLowContrastLibCacheDir = '/scratch/xh2906/librarySCI_runs/sixd_prefAngle_cosonly_masked_bs8192_steps40_rmse1_rel3_low005_mono0p1_regen_masked_newsplit_tanh3_mixedmask_S30_C100_I140_randomrowsplit_drcorrected_20260328_231201_lowcontrast_allfix_test_20260423_095028/hpc_runtime_bundle/equilibrium_14_lowcontrast_precomputed_nnorder_20260426_105149';
SpecialLowContrastPermRoot = '/scratch/xh2906/librarySCI_runs/sixd_prefAngle_cosonly_masked_bs8192_steps40_rmse1_rel3_low005_mono0p1_regen_masked_newsplit_tanh3_mixedmask_S30_C100_I140_randomrowsplit_drcorrected_20260328_231201_lowcontrast_allfix_test_20260423_095028/hpc_runtime_bundle/equilibrium_3_lowcontrast_allperm_20260426_113729';
if ~exist('LibCacheDirOverride', 'var') || isempty(LibCacheDirOverride)
    LibCacheDirOverride = DefaultFull70LibCacheDir;
end
if ~exist('LowContrastLibCacheDirOverride', 'var') || isempty(LowContrastLibCacheDirOverride)
    LowContrastLibCacheDirOverride = DefaultLowContrastLibCacheDir;
end
LowContrastCacheAngles = [0 3.75 7.50 11.25 15.00 18.75 22.50];
SpecialLowContrastLibCacheDir = '';
if abs(Contr - 5) < 1e-8 && abs(Angle - 18.75) < 1e-8
    SpecialLowContrastLibCacheDir = fullfile(SpecialLowContrastPermRoot, 'angle_18.75_contr5_order_2_1_3_4');
elseif abs(Contr - 14) < 1e-8 && abs(Angle - 22.50) < 1e-8
    SpecialLowContrastLibCacheDir = fullfile(SpecialLowContrastPermRoot, 'angle_22.50_contr14_order_3_2_1_4');
end
UseSpecialLowContrastLibCache = ~isempty(SpecialLowContrastLibCacheDir) && isfolder(SpecialLowContrastLibCacheDir);
if exist('Extra50Case','var') && Extra50Case, UseSpecialLowContrastLibCache = false; end
UseLowContrastLibCache = any(abs(Contr - [5 14]) < 1e-8) && ...
    any(abs(Angle - LowContrastCacheAngles) < 1e-8) && ...
    isfolder(LowContrastLibCacheDirOverride);

LibCacheData = [];
LibCacheDirUse = '';
cacheFile = '';
LibCacheCandidateDirs = {};
if exist('LibCacheDirOverride', 'var') && ~isempty(LibCacheDirOverride) && isfolder(LibCacheDirOverride)
    LibCacheCandidateDirs{end+1} = LibCacheDirOverride; %#ok<SAGROW>
end
if UseSpecialLowContrastLibCache
    LibCacheCandidateDirs{end+1} = SpecialLowContrastLibCacheDir; %#ok<SAGROW>
end
if UseLowContrastLibCache
    LibCacheCandidateDirs{end+1} = LowContrastLibCacheDirOverride; %#ok<SAGROW>
end
for CacheDirIdx = 1:numel(LibCacheCandidateDirs)
    candidateDir = LibCacheCandidateDirs{CacheDirIdx};
    candidateEig = fullfile(candidateDir, sprintf('eig_16HC_6DprefAngle_input_contr%d_angle_%.2f.mat', Contr, Angle));
    candidateLibOnly = fullfile(candidateDir, sprintf('libonly_16HC_6DprefAngle_input_contr%d_angle_%.2f.mat', Contr, Angle));
    if isfile(candidateEig)
        LibCacheDirUse = candidateDir;
        cacheFile = candidateEig;
        break;
    elseif isfile(candidateLibOnly)
        LibCacheDirUse = candidateDir;
        cacheFile = candidateLibOnly;
        break;
    end
end
UseLibCache = ~isempty(cacheFile);
if UseLibCache
    fprintf('Using cached library equilibrium: %s\n', cacheFile);
    LibCacheData = load(cacheFile, 'LDEEquv', 'LDEEquv_rotlib', 'S_map_fulllib', 'C_map_fulllib', 'I_map_fulllib');
    if ~isfield(LibCacheData, 'LDEEquv')
        error('Cached library file missing LDEEquv: %s', cacheFile);
    end
end

%% block 13

L4EmeshXAll = cell(size(DomList));
L4ImeshYAll = cell(size(DomList));
LDEFrfuncAll = cell(size(DomList));
for DomInd = 1:length(DomList)
    domName = DomList{DomInd};
    domPattern = sprintf('Func200%s-%s-Ang*-SF%.1fTF%s%s-%s*.mat', ...
        DataPt, Comment, lgnSF, lgnTFLabel, ContrStr, domName);
    domFiles = dir(fullfile(SaveToFolder, domPattern));
    if isempty(domFiles)
        error('No precomputed file candidates found for pattern: %s', domPattern);
    end

    bestId = 0;
    bestDelta = inf;
    bestPart = false;
    bestAngle = NaN;
    for fId = 1:numel(domFiles)
        fName = domFiles(fId).name;
        tok = regexp(fName, '-Ang([0-9]+(?:\.[0-9]+)?)-SF', 'tokens', 'once');
        if isempty(tok)
            continue;
        end
        angNow = str2double(tok{1});
        delta = abs(angNow - Angle);
        isPart = contains(fName, '_part.mat');

        if delta < bestDelta - 1e-9 || (abs(delta-bestDelta) <= 1e-9 && isPart && ~bestPart)
            bestId = fId;
            bestDelta = delta;
            bestPart = isPart;
            bestAngle = angNow;
        end
    end

    if bestId == 0
        error('Failed to parse any angle token from files matching: %s', domPattern);
    end
    if bestDelta > 0.26
        error('Closest file angle too far from target: target=%.2f, closest=%.2f, file=%s', ...
            Angle, bestAngle, domFiles(bestId).name);
    end

    useFile = fullfile(domFiles(bestId).folder, domFiles(bestId).name);
    fprintf('Loading Dom=%s target angle %.2f from file: %s\n', domName, Angle, domFiles(bestId).name);
    FuncTemp = load(useFile, 'LDEFrfunc','L4EmeshX','L4ImeshY');
    L4EmeshXAll{DomInd} = FuncTemp.L4EmeshX;
    L4ImeshYAll{DomInd} = FuncTemp.L4ImeshY;
    LDEFrfuncAll{DomInd} = FuncTemp.LDEFrfunc;
end

%% block 14
%    N_HCOut = 4; PixNumOut = N_HCOut^2*NPixX*NPixY;
LGNlist = 5; L6list = 33; % Why do we have L6list here?
OD_SMapModi = OD_SMap;

%% block 15
%OD_SMapModi = OD_SMap(1:2*n_S_HC,1:2*n_S_HC);
if MonocuFlag
    OD_SMapModi(n_S_HC+1:2*n_S_HC,1:end) = 5;
    SaveStr = ['Mono' Comment];
else
    SaveStr = ['Bino' Comment];
end

%% block 16
L6ConvSig = 0.75;
L6ConvTruc = 1.5/(L6ConvSig);
Kersize = floor(2*L6ConvSig*L6ConvTruc);
if Kersize<=0
    Kersize= 2; % at least 1
end
[x,y] = meshgrid(1:Kersize,1:Kersize);
c = Kersize/2;
exponent = ((x-c).^2+(y-c).^2)/(2*L6ConvSig^2);
L6Kernel = exp(-exponent);
L6Kernel(L6Kernel<exp(-(L6ConvTruc*1.0)^2/2))=0; % truncate
L6Kernel = L6Kernel + ...
    L6Kernel(end:-1:1,:) + ...
    L6Kernel(:,end:-1:1) + ...
    L6Kernel(end:-1:1,end:-1:1);%symmetrize
L6Kernel = L6Kernel/sum(L6Kernel,'all'); % normalize
L6Kernel = L6Kernel/((sum(L6Kernel,'all') - L6Kernel(2,2))/(1-0.3));
L6Kernel(2,2) = 0.3;

%% block 17
N_HCs = 3; % shrink to 2*2
FigOn = false;
LGNsmear = n_S_HC*0.2;
TruncLGN = 1;
LGNFilt_Grating = SpatialGaussianFilt_my(OD_SMapModi,...
    N_HCs,n_S_HC,LGNsmear,TruncLGN,FigOn);

if ~MonocuFlag
    LGNFilt_Grating = [LGNFilt_Grating,zeros(size(LGNFilt_Grating,1),1)];
end

PixLGNCtgr = LGNIndSpat_Rec(...
    LGNFilt_Grating,1:LGNlist,NnSPixel,N_HC,N_HCOutX,N_HCOutY,NPixX,NPixY,true);

% symmetrize !! This is actually trick, so I do it ad hoc
PixLGNCtgr = Ocu_LGNL6symm(PixLGNCtgr,N_HCOutX,N_HCOutY,NPixX,NPixY);

%% block 18
if ~exist('BGFlagOverride', 'var') || isempty(BGFlagOverride)
    BGFlag = false;
else
    BGFlag = logical(BGFlagOverride);
end
if BGFlag
    % PixL6Ctgr(:,end) = 1; PixL6Ctgr(:,1:end-1) = 0;
    PixLGNCtgr(:,end) = 1; PixLGNCtgr(:,1:end-1) = 0;
end

%% block 19
MirInd = logical(MirInd);
CtgrOrder = [2,1;3,4]; % for lower left HC
CtgrOrderUse = rot90(CtgrOrder,RotInd);
if MirInd
    %flipDim = mod(RotInd,2)+1; %1 for col 2 ofr row
    CtgrOrderUse = flip(CtgrOrderUse,1);
end
CtgrOrderReadout = ...
    [CtgrOrderUse(1,2);
    CtgrOrderUse(1,1);
    CtgrOrderUse(2,1);
    CtgrOrderUse(2,2)];
PixLGNCtgrLib = PixLGNCtgr;
%    for Ctgr = 1:4
%         PixL6Ctgr(:,Ctgr) = ...
%             HCRot_Rec(PixL6Ctgr(:,Ctgr),RotInd,N_HCOutX,N_HCOutY,NPixX,NPixY,MirInd);
%         PixLGNCtgr(:,Ctgr) = ...
%             HCRot_Rec(PixLGNCtgr(:,Ctgr),RotInd,N_HCOutX,N_HCOutY,NPixX,NPixY,MirInd);
%    end

% let's implement something cheap now for 45 deg
% Keep the NN/surrogate path on the original LGN-category logic.
% Only the library output channels are reassigned by CtgrOrderReadout.

%% block 20
ParaEx = TestExSeq(TestId); ParaEy = TestEySeq(TestId);
ParaIx = TestIxSeq(TestId); ParaIy = TestIySeq(TestId);

C_SS_mean = sparse(AveSpatKer_Rec(C_SS_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaEx,ParaEy));
C_CS_mean = sparse(AveSpatKer_Rec(C_CS_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaEx,ParaEy));
C_IS_mean = sparse(AveSpatKer_Rec(C_IS_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaEx,ParaEy));
C_SC_mean = sparse(AveSpatKer_Rec(C_SC_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaEx,ParaEy));
C_CC_mean = sparse(AveSpatKer_Rec(C_CC_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaEx,ParaEy));
C_IC_mean = sparse(AveSpatKer_Rec(C_IC_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaEx,ParaEy));
C_SI_mean = sparse(AveSpatKer_Rec(C_SI_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaIx,ParaIy));
C_CI_mean = sparse(AveSpatKer_Rec(C_CI_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaIx,ParaIy));
C_II_mean = sparse(AveSpatKer_Rec(C_II_Pixel_Us,N_HC,...
    N_HCOutX,N_HCOutY,NPixX,NPixY,ParaIx,ParaIy));

Conn_Reduc_TestS = 6; Conn_Reduc_TestC = 4;
Reduc_S = Conn_Reduc_TestS*(1-CplxR); Reduc_C = Conn_Reduc_TestC*CplxR;
P_SS_rate = (N_SS-Reduc_S)/N_SS; C_SS_meanU = C_SS_mean * P_SS_rate;
P_SC_rate = (N_SC-Reduc_C)/N_SC; C_SC_meanU = C_SC_mean * P_SC_rate;
P_CS_rate = (N_CS-Reduc_S)/N_CS; C_CS_meanU = C_CS_mean * P_CS_rate;
P_CC_rate = (N_CC-Reduc_C)/N_CC; C_CC_meanU = C_CC_mean * P_CC_rate;

%% block 21
%     EKp.Thrsld = 50; EKp.Highist = [200,150]; EKp.HardBound = 200; EKp.Slope = 0;
%
%     IKp.Thrsld = 70; IKp.Highist = [100,95];  IKp.HardBound = 120; IKp.Slope = 1; % 1; 0.9(95); 0.8(9)

%% block 22
tic
ICUse = ICTestAll{TestId}; %
IniTest = ICUse;


IniTest.S(:) = 2.5;
IniTest.C(:) = 8;
IniTest.I(:) = 18;
%IniTest.S = symmHCs(ICUse.S,N_HCOut,NPixX,NPixY);
%IniTest.C = symmHCs(ICUse.C,N_HCOut,NPixX,NPixY);
%IniTest.I = symmHCs(ICUse.I,N_HCOut,NPixX,NPixY);

%% block 23
EpocTest=300;
Contruse = Contr * ones(1600, 1);
Orientationuse = angle * ones(1600, 1);
tic
[LDEOutAll{TestId},LDEIL2Diff{TestId},L2DiffNormNeib{TestId},~,~,~,...
    NANFlag(TestId),OperatingPoint] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle(...
    PixLGNCtgr, L6Kernel, IniTest, p, L6pars{L6parId}, EpocTest,...
    C_SS_meanU,C_CS_meanU,C_IS_mean,...
    C_SC_meanU,C_CC_meanU,C_IC_mean,...
    C_SI_mean, C_CI_mean, C_II_mean,...
    L4SEp, L4SIp, ...
    L4CEp, L4CIp, ...
    L4IEp, L4IIp, ...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll,Contruse,Orientationuse,...
    N_HCOutY,NPixX,NPixY,Isaturation,'xn',EKpUse,IKpUse);
LDEsigmoid=LDEOutAll{TestId}{end};
toc

%% block 24


if UseLibCache
    LDEOut{TestId} = [];
    LDEEquv = LibCacheData.LDEEquv; % reuse cached library equilibrium
else
    [LDEOut{TestId},LDEIL2Diff{TestId},L2DiffNormNeib{TestId},~,~,FuncUse,...
        NANFlag(TestId)] = ...
        LDEIteration_135FuncMain_CombDom_RealLGNL6(...
        PixLGNCtgr,L6Kernel,IniTest,p,L6pars{L6parId}, EpocTest,...
        C_SS_meanU,C_CS_meanU,C_IS_mean,...
        C_SC_meanU,C_CC_meanU,C_IC_mean,...
        C_SI_mean, C_CI_mean, C_II_mean,...
        L4SEp, L4SIp, ...
        L4CEp, L4CIp, ...
        L4IEp, L4IIp, ...
        L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll,...
        CtgrOrderReadout, angle, Contr, N_HCOutY,NPixX,NPixY,Isaturation,'xn',EKpUse,IKpUse)  ;

    LDEEquv = LDEOut{TestId}{end}; % this record the final NESS state
end

% Evaluate the full library once at the final NN operating state. This is a
% matched local-response target, not the independently converged library state.
[OperatingTargetTrace,~,~,~,~,~,OperatingTargetNAN] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6(...
    PixLGNCtgr,L6Kernel,LDEsigmoid,1,L6pars{L6parId},1,...
    C_SS_meanU,C_CS_meanU,C_IS_mean,...
    C_SC_meanU,C_CC_meanU,C_IC_mean,...
    C_SI_mean,C_CI_mean,C_II_mean,...
    L4SEp,L4SIp,L4CEp,L4CIp,L4IEp,L4IIp,...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll,...
    CtgrOrderReadout,angle,Contr,N_HCOutY,NPixX,NPixY,...
    Isaturation,'xn',EKpUse,IKpUse);
if OperatingTargetNAN
    error('Full-library operating-point target produced NaN.');
end
OperatingTarget = OperatingTargetTrace{end};

% Capture the model-query inputs induced by the independently converged
% full-library state. The MLP output from this one-step call is ignored;
% only the pre-response L4E/L4I/L6 operating point is used for training.
[~,~,~,~,~,~,OperatingPointLibNAN,OperatingPointLib] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle(...
    PixLGNCtgr,L6Kernel,LDEEquv,1,L6pars{L6parId},1,...
    C_SS_meanU,C_CS_meanU,C_IS_mean,...
    C_SC_meanU,C_CC_meanU,C_IC_mean,...
    C_SI_mean,C_CI_mean,C_II_mean,...
    L4SEp,L4SIp,L4CEp,L4CIp,L4IEp,L4IIp,...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll,Contruse,Orientationuse,...
    N_HCOutY,NPixX,NPixY,Isaturation,'xn',EKpUse,IKpUse);
if OperatingPointLibNAN
    error('Library-equilibrium operating-point capture produced NaN.');
end
OperatingTargetLib = LDEEquv;

    % LDEEquv = ICUse;  % this is to compare it with the precomputed NESS state, that is ZC's result.

%% block 25


    if ~NANFlag(TestId)
    FinalEquv = LDEEquv;
    for EpcInd = 1:EpocTest+1
        LDEPertRslt = LDEOutAll{TestId}{EpcInd};
        % LDEPertRslt = LDEICOut;
        SDiff = LDEPertRslt.S - FinalEquv.S;
        SDiffHC = reshape(SDiff,N_HCOutY*NPixY,N_HCOutX*NPixX);
        SDiff = reshape(SDiffHC(1:NPixY,1:NPixX),NPixY*NPixX,1);
        CDiff = LDEPertRslt.C - FinalEquv.C;
        CDiffHC = reshape(CDiff,N_HCOutY*NPixY,N_HCOutX*NPixX);
        CDiff = reshape(CDiffHC(1:NPixY,1:NPixX),NPixY*NPixX,1);
        IDiff = LDEPertRslt.I - FinalEquv.I;
        IDiffHC = reshape(IDiff,N_HCOutY*NPixY,N_HCOutX*NPixX);
        IDiff = reshape(IDiffHC(1:NPixY,1:NPixX),NPixY*NPixX,1);

        L2Diff_OneStepAll(TestId,EpcInd) = ...
            sqrt(sum((SDiff*(1-CplxR)+CDiff*CplxR).^2,'all'));%sqrt(sum([SDiff;CDiff;IDiff].^2));
        DiffVecAll(TestId,EpcInd,:) = [SDiff;CDiff;IDiff];

        aE = 32 / (32 + 8);   % 0.8
        aI = 8  / (32 + 8);   % 0.2
        wC = 0.3077; 
        err_ode_true_all(TestId,EpcInd) = HC_norm_diff(FinalEquv.S, FinalEquv.C, FinalEquv.I, ...
                                LDEPertRslt.S, LDEPertRslt.C, LDEPertRslt.I, ...
                                wC, aE, aI);


    end
end

RunTime = toc;
sprintf('RunTime = %.2f',RunTime)


%% block 26
N_HCOut_full = 4;                   % full network
LDEEquv_rot = HCRot(LDEOutAll{TestId}{end}, RotInd, N_HCOut_full, NPixX, NPixY, MirInd);
if UseLibCache
    if isfield(LibCacheData, 'LDEEquv_rotlib')
        LDEEquv_rotlib = LibCacheData.LDEEquv_rotlib;
    else
        LDEEquv_rotlib = HCRot(LDEEquv, RotInd, N_HCOut_full, NPixX, NPixY, MirInd);
    end
else
    LDEEquv_rotlib = HCRot(LDEOut{TestId}{end}, RotInd, N_HCOut_full, NPixX, NPixY, MirInd);
end

S_map_full = reshape(LDEEquv_rot.S, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
C_map_full = reshape(LDEEquv_rot.C, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
I_map_full = reshape(LDEEquv_rot.I, NPixX * N_HCOut_full, NPixY * N_HCOut_full);

if UseLibCache && isfield(LibCacheData, 'S_map_fulllib') && isfield(LibCacheData, 'C_map_fulllib') && isfield(LibCacheData, 'I_map_fulllib')
    S_map_fulllib = LibCacheData.S_map_fulllib;
    C_map_fulllib = LibCacheData.C_map_fulllib;
    I_map_fulllib = LibCacheData.I_map_fulllib;
else
    S_map_fulllib = reshape(LDEEquv_rotlib.S, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
    C_map_fulllib = reshape(LDEEquv_rotlib.C, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
    I_map_fulllib = reshape(LDEEquv_rotlib.I, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
end

% Plot
comparisonequi = figure;

% --- Compute shared color limits for each column ---

% Column 1: S maps
clim_S = [ ...
    min([S_map_full(:); S_map_fulllib(:)]), ...
    max([S_map_full(:); S_map_fulllib(:)]) ];

% Column 2: C maps
clim_C = [ ...
    min([C_map_full(:); C_map_fulllib(:)]), ...
    max([C_map_full(:); C_map_fulllib(:)]) ];

% Column 3: I maps
clim_I = [ ...
    min([I_map_full(:); I_map_fulllib(:)]), ...
    max([I_map_full(:); I_map_fulllib(:)]) ];


% --- Plot with shared colorbars per column ---

subplot(2,3,1);
imagesc(S_map_full);
caxis(clim_S);
colorbar;
axis equal tight;

subplot(2,3,2);
imagesc(C_map_full);
caxis(clim_C);
colorbar;
axis equal tight;
title('NN');

subplot(2,3,3);
imagesc(I_map_full);
caxis(clim_I);
colorbar;
axis equal tight;


subplot(2,3,4);
imagesc(S_map_fulllib);
caxis(clim_S);
colorbar;
axis equal tight;

subplot(2,3,5);
imagesc(C_map_fulllib);
caxis(clim_C);
colorbar;
axis equal tight;
title('Library');

subplot(2,3,6);
imagesc(I_map_fulllib);
caxis(clim_I);
colorbar;
axis equal tight;


sgtitle(sprintf('6D Pref-Angle Equilibrium Comparison (Angle = %.2f deg, Contr = %d)\nHC norm = %.3f', ...
    Angle, Contr, err_ode_true_all(TestId,end)));
if exist('SaveBaseDirOverride', 'var') && ~isempty(SaveBaseDirOverride)
    saveFolder = SaveBaseDirOverride;
else
    saveFolder = ['/Users/xialeihuang/Desktop/Neuroscience_Project/' ...
        'Spectral_Analysis/matlab-inserting_into_CG_model/Complete_Code_for_Paper3/' ...
        'NYU-Vision-2Drive-main/Figures/spectral_analysis_eigenvalue_eigenvectors/'];
end
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
filename = fullfile(saveFolder, sprintf('eig_16HC_6DprefAngle_input_contr%d_angle_%.2f.mat', ...
    Contr, Angle));
figFilename = fullfile(saveFolder, ...
    sprintf('Equilibrium_Comparison_6DprefAngle_angle_%.2f_contr%d.png', Angle, Contr));

if ~exist('SavePngOverride', 'var') || SavePngOverride
    saveas(comparisonequi, figFilename);
end

if exist('SaveFigOverride', 'var') && SaveFigOverride
    savefig(comparisonequi, strrep(figFilename, '.png', '.fig'));
end
save(filename, 'Angle', 'Contr', 'LDEsigmoid', 'LDEEquv', 'LDEEquv_rot', ...
    'LDEEquv_rotlib', 'S_map_full', 'C_map_full', 'I_map_full', ...
    'S_map_fulllib', 'C_map_fulllib', 'I_map_fulllib', 'err_ode_true_all', 'lgnTF', 'lgnSF', 'SaveToFolder', 'OperatingPoint', 'OperatingTarget', 'OperatingPointLib', 'OperatingTargetLib');

%% block 27
    
end
end
