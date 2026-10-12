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

CurrentFolder = '/scratch/xh2906/NYU-Vision-2Drive-main'
addpath(CurrentFolder)
addpath([CurrentFolder '/Utils'])
addpath([CurrentFolder '/Data'])
if exist('RuntimeBundleDirOverride', 'var') && ~isempty(RuntimeBundleDirOverride)
    addpath(RuntimeBundleDirOverride, '-begin');
end
SaveToFolder = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/Paper3PlotingData/']; % V1D2
if exist('LibrarySourceDirOverride', 'var') && ~isempty(LibrarySourceDirOverride)
    SaveToFolder = LibrarySourceDirOverride;
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
lgnSF = 2.5; lgnTF = 10;

ExpTex = 'CtrlL4';
Comment = ['L6Real' ExpTex];

L6pars = {};
L6pars{3} = {2.5,3,[28; 40; 50],[62.5; 84; 96],'quadratic'};
L6pars{1} = {2.5,3,62.5,86,86};
L6pars{2} = {2.5,3,62.5,86,80.4,106,90,142.2, 102, 166.7, 108, 210, 118}; % 142.2; 166.7

L6ParamRawUse = {2.5,3,62.5,86,80.4,106,90,142.2,102,166.7,108,210,118,{[0 15 29],[0,34.0],[28 32 50],[69 85]}};
L6C1Grid = 0:0.25:120;
L6ParamUse = {2.5,3,{'c1smooth',L6ParamRawUse,L6C1Grid}};
L6pars{5} = L6ParamUse;


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
IKp3.Mode = 'multisigmoid'; % or 'linear'
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
    AngleList = {'0.00','3.75','7.50','11.25','15.00','18.75','22.50'};
else
    AngleList = AngleListOverride;
end
DomList = {'Small'};

% Library-only merged dataset uses the explicit DR100/DS10 contrast grid.
ContrListAll = [14 19 24 33 38 42 47 52 61 66 71 80 89 95 100];
if ~exist('ContrListOverride', 'var') || isempty(ContrListOverride)
    RequestedContrList = ContrListAll;
else
    RequestedContrList = ContrListOverride;
end
if ~all(ismember(RequestedContrList, ContrListAll))
    error('One or more requested contrasts are outside the available full old-form library grid.');
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
funcAnglePattern = sprintf('Func200%s-%s-Ang*-SF%.1fTF%d%s-%s*.mat', ...
    DataPt, Comment, lgnSF, lgnTF, ContrStr, DomList{1});
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

UseLibCache = false;
LibCacheData = [];
LibCacheDirUse = '';

%% block 13
% Library-only exact source lookup for angle-22.5 full no-downsample old-form files.
% No cached equilibrium, no NN, and no alternate part/DS10 file suffix.
L4EmeshXAll = cell(size(DomList));
L4ImeshYAll = cell(size(DomList));
LDEFrfuncAll = cell(size(DomList));
for DomInd = 1:length(DomList)
    domName = DomList{DomInd};
    sourceTag = 'fullnods_smooth2d';
    AngPrintOld = regexprep(regexprep(sprintf('%.2f', Angle), '0+$', ''), '\.$', '');
    if Contr == 100
        ContrStrOld = '';
    else
        ContrStrOld = sprintf('_Contr%d', Contr);
    end
    useFile = fullfile(SaveToFolder, sprintf('Func200%s-%s-Ang%s-SF%.1fTF%d%s-%s.mat', ...
        DataPt, Comment, AngPrintOld, lgnSF, lgnTF, ContrStrOld, domName));
    if ~isfile(useFile)
        AngPrintLegacy = sprintf('%.1f', Angle);
        useFileLegacy = fullfile(SaveToFolder, sprintf('Func200%s-%s-Ang%s-SF%.1fTF%d%s-%s.mat', ...
            DataPt, Comment, AngPrintLegacy, lgnSF, lgnTF, ContrStrOld, domName));
        AngPrintCompact = regexprep(regexprep(sprintf('%.2f', Angle), '0+$', ''), '\.$', '');
        useFileCompact = fullfile(SaveToFolder, sprintf('Func200%s-%s-Ang%s-SF%.1fTF%d%s-%s.mat', ...
            DataPt, Comment, AngPrintCompact, lgnSF, lgnTF, ContrStrOld, domName));
        if isfile(useFileLegacy)
            useFile = useFileLegacy;
        elseif isfile(useFileCompact)
            useFile = useFileCompact;
        else
            error('Required exact %s library file not found: %s or %s or %s', sourceTag, useFile, useFileLegacy, useFileCompact);
        end
    end
    fprintf('Loading Dom=%s exact source=%s angle %.2f contrast %d from file: %s\n', ...
        domName, sourceTag, Angle, Contr, useFile);
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
% Match eigenvalue_5D_mlp_contrast_orientation_loop_only_equilibrium.mlx:
% reorder the LGN spatial category columns before the library iteration.
PixLGNCtgr = PixLGNCtgr(:, [CtgrOrderReadout', 5]);

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
% NN/surrogate iteration intentionally removed for library-only validation.

%% block 24
[LDEOut{TestId},LDEIL2Diff{TestId},L2DiffNormNeib{TestId},~,~,FuncUse, ...
    NANFlag(TestId)] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6_noreorder(...
    PixLGNCtgr,L6Kernel,IniTest,p,L6pars{L6parId}, EpocTest,...
    C_SS_meanU,C_CS_meanU,C_IS_mean,...
    C_SC_meanU,C_CC_meanU,C_IC_mean,...
    C_SI_mean, C_CI_mean, C_II_mean,...
    L4SEp, L4SIp, ...
    L4CEp, L4CIp, ...
    L4IEp, L4IIp, ...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll,...
    [1;2;3;4], angle, Contr, N_HCOutY,NPixX,NPixY,Isaturation,'xn',EKpUse,IKpUse);
LDEEquv = LDEOut{TestId}{end};

%% block 25
RunTime = toc;
fprintf('RunTime = %.2f\n', RunTime);

%% block 26
N_HCOut_full = 4;
LDEEquv_rotlib = HCRot(LDEEquv, RotInd, N_HCOut_full, NPixX, NPixY, MirInd);
S_map_fulllib = reshape(LDEEquv_rotlib.S, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
C_map_fulllib = reshape(LDEEquv_rotlib.C, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
I_map_fulllib = reshape(LDEEquv_rotlib.I, NPixX * N_HCOut_full, NPixY * N_HCOut_full);
wC = 0.3077;
E_map_fulllib = (1 - wC) * S_map_fulllib + wC * C_map_fulllib;

if exist('SaveBaseDirOverride', 'var') && ~isempty(SaveBaseDirOverride)
    saveFolder = SaveBaseDirOverride;
else
    saveFolder = fullfile(CurrentFolder, 'Figures', 'library_only_70_fullnods_oldform');
end
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
filename = fullfile(saveFolder, sprintf('libonly_16HC_6DprefAngle_input_contr%d_angle_%.2f.mat', Contr, Angle));
figFilename = fullfile(saveFolder, sprintf('LibraryOnly_6DprefAngle_angle_%.2f_contr%d.png', Angle, Contr));

comparisonequi = figure('Visible','off','Position',[100 100 1800 430]);
subplot(1,4,1); imagesc(S_map_fulllib); colorbar; axis equal tight; title('S library');
subplot(1,4,2); imagesc(C_map_fulllib); colorbar; axis equal tight; title('C library');
subplot(1,4,3); imagesc(I_map_fulllib); colorbar; axis equal tight; title('I library');
subplot(1,4,4); imagesc(E_map_fulllib); colorbar; axis equal tight; title('E library');
sgtitle(sprintf('Library-only full no-downsample smooth2d equilibrium (Angle = %.2f deg, Contr = %d, source = %s)', ...
    Angle, Contr, sourceTag));
saveas(comparisonequi, figFilename);
close(comparisonequi);

save(filename, 'Angle', 'Contr', 'sourceTag', 'LDEEquv', 'LDEEquv_rotlib', ...
    'S_map_fulllib', 'C_map_fulllib', 'I_map_fulllib', 'E_map_fulllib', ...
    'FuncUse', 'NANFlag', 'RunTime');

%% block 27
    
end
end
