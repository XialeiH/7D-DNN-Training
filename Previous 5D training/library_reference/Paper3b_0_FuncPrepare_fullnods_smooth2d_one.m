function Paper3b_0_FuncPrepare_fullnods_smooth2d_one(Contr, Angle, SaveFolder)
% Re-aggregate existing full no-downsample raw LIF outputs with proper 2D L4E/L4I smoothing.
% This does not rerun LIF simulation. It writes old-form Func200...mat files into SaveFolder.
if nargin < 3 || isempty(SaveFolder)
    error('Usage: Paper3b_0_FuncPrepare_fullnods_smooth2d_one(Contr, Angle, SaveFolder)');
end
CurrentFolder = '/scratch/xh2906/NYU-Vision-2Drive-main';
addpath(CurrentFolder)
addpath([CurrentFolder '/Utils'])
addpath([CurrentFolder '/Data'])
DataFolder2 = [CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/25Function_binocular_realLGN/'];
addpath(DataFolder2)
addpath([CurrentFolder '/Data/Paper2_NetworkTuning/Fig1V4/'])
if ~exist(SaveFolder, 'dir')
    mkdir(SaveFolder);
end
DataPt = 'V4D2';
S = load(fullfile(CurrentFolder, 'Data', 'Paper2_NetworkTuning', 'Fig1V4', sprintf('AllMFPixPara_Paper2TuneFig1%s.mat',DataPt))); %#ok<NASGU>
if Contr == 100
    ContrStr = '';
else
    ContrStr = sprintf('_Contr%d', Contr);
end
lgnSF = 2.5;
lgnTF = 10;
AngPrint = regexprep(regexprep(sprintf('%.2f', Angle), '0+$', ''), '\.$', '');
if abs(Angle*2 - round(Angle*2)) < 1e-8
    AngPrintOut = sprintf('%.1f', Angle);
else
    AngPrintOut = AngPrint;
end
DomFlag = 'S';
DomPrint = 'Small';
ExpTex = 'CtrlL4';
Comment = ['L6Real' ExpTex];
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

OutPath = fullfile(SaveFolder, sprintf('Func200%s-%s-Ang%s-SF%.1fTF%d%s-%s.mat', DataPt, Comment, AngPrintOut, lgnSF, lgnTF, ContrStr, DomPrint));
fprintf('Generating smooth2d full aggregate %s\n', OutPath);

FullPath0 = findFullRaw(DataFolder2, DataPt, AngPrint, Angle, 1, 1, lgnSF, lgnTF, ContrStr, L6up, L6low, ExpTex, DomFlag);
S0 = load(FullPath0, 'L4ERcrd', 'L4IRcrd', 'f_EnIOut');
L4ERcrd = S0.L4ERcrd(:);
L4IRcrd = S0.L4IRcrd(:);
a1 = numel(unique(L4ERcrd));
a2 = floor(numel(L4ERcrd) / max(a1,1));
if a1*a2 ~= numel(L4ERcrd) || a2 < 1
    error('Full raw coordinates are not a rectangular mesh for Angle=%.2f Contr=%d.', Angle, Contr);
end
L4EPlot = reshape(L4ERcrd, a2, a1);
L4IPlot = reshape(L4IRcrd, a2, a1);
LDEFrVecS = nan(numel(L4ERcrd), LGNlist, L6list);
LDEFrVecC = nan(numel(L4ERcrd), LGNlist, L6list);
LDEFrVecI = nan(numel(L4ERcrd), LGNlist, L6list);

for LGNCtgr = 1:LGNlist
    fprintf('  LGNCtgr %d\n', LGNCtgr);
    for L6Ctgr = 1:L6list
        FullPath = findFullRaw(DataFolder2, DataPt, AngPrint, Angle, LGNCtgr, L6Ctgr, lgnSF, lgnTF, ContrStr, L6up, L6low, ExpTex, DomFlag);
        D = load(FullPath, 'f_EnIOut', 'L4ERcrd', 'L4IRcrd');
        if numel(D.L4ERcrd) ~= numel(L4ERcrd) || any(abs(D.L4ERcrd(:)-L4ERcrd)>1e-9) || any(abs(D.L4IRcrd(:)-L4IRcrd)>1e-9)
            error('Coordinate mismatch in %s', FullPath);
        end
        FrLDE = nan(numel(D.f_EnIOut), LIFN);
        for LDEInd = 1:numel(D.f_EnIOut)
            if isempty(D.f_EnIOut{LDEInd}) || size(D.f_EnIOut{LDEInd},2) < 5
                continue
            end
            nT = size(D.f_EnIOut{LDEInd},2);
            cStart = max(1, floor(nT/2));
            FrLDE(LDEInd,:) = (mean(D.f_EnIOut{LDEInd}(:,cStart:nT),2))';
        end
        FrLDE(FrLDE<0) = eps;
        [LDEFrVecS(:,LGNCtgr,L6Ctgr), ~] = LDEPlotRateFunc_RealLGN_NoPlot(L4EPlot,L4IPlot,FrLDE,a1,a2,LGNCtgr,L6Ctgr,'S',smth,plotFlag,realLGN,[rowsmWin, colswWin]);
        [LDEFrVecC(:,LGNCtgr,L6Ctgr), ~] = LDEPlotRateFunc_RealLGN_NoPlot(L4EPlot,L4IPlot,FrLDE,a1,a2,LGNCtgr,L6Ctgr,'C',smth,plotFlag,realLGN,[rowsmWin, colswWin]);
        [LDEFrVecI(:,LGNCtgr,L6Ctgr), ~] = LDEPlotRateFunc_RealLGN_NoPlot(L4EPlot,L4IPlot,FrLDE,a1,a2,LGNCtgr,L6Ctgr,'I',smth,plotFlag,realLGN,[rowsmWin, colswWin]);
    end
end

L4EfuncX = linspace(min(L4ERcrd), max(L4ERcrd), 400);
L4IfuncY = linspace(min(L4IRcrd), max(L4IRcrd), 300);
[L4EmeshX, L4ImeshY] = meshgrid(L4EfuncX, L4IfuncY);
ItplType = 'Linear';
LDEFrfunc = struct('S',{cell(LGNlist,L6list)},'C',{cell(LGNlist,L6list)},'I',{cell(LGNlist,L6list)});
for LGNCtgr = 1:LGNlist
    for L6Ctgr = 1:L6list
        vecS = LDEFrVecS(:,LGNCtgr,L6Ctgr); validS = isfinite(vecS);
        LDEFrfunc.S{LGNCtgr,L6Ctgr} = griddata(L4ERcrd(validS), L4IRcrd(validS), vecS(validS), L4EmeshX, L4ImeshY, ItplType);
        vecC = LDEFrVecC(:,LGNCtgr,L6Ctgr); validC = isfinite(vecC);
        LDEFrfunc.C{LGNCtgr,L6Ctgr} = griddata(L4ERcrd(validC), L4IRcrd(validC), vecC(validC), L4EmeshX, L4ImeshY, ItplType);
        vecI = LDEFrVecI(:,LGNCtgr,L6Ctgr); validI = isfinite(vecI);
        LDEFrfunc.I{LGNCtgr,L6Ctgr} = griddata(L4ERcrd(validI), L4IRcrd(validI), vecI(validI), L4EmeshX, L4ImeshY, ItplType);
    end
end
if abs(Angle - 22.5) < 1e-8
    LDEFrfuncUse.S = FuncSwap(LDEFrfunc.S,[1,2,3,4;4,3,2,1]);
    LDEFrfuncUse.C = FuncSwap(LDEFrfunc.C,[1,2,3,4;4,3,2,1]);
    LDEFrfuncUse.I = FuncSwap(LDEFrfunc.I,[1,2,3,4;4,3,2,1]);
    LDEFrfunc = LDEFrfuncUse;
end
save(OutPath, 'LDEFrfunc', 'L4EmeshX', 'L4ImeshY')
fprintf('Saved %s\n', OutPath);
end

function FullPath = findFullRaw(DataFolder2, DataPt, AngPrint, Angle, LGNCtgr, L6Ctgr, lgnSF, lgnTF, ContrStr, L6up, L6low, ExpTex, DomFlag)
LGNStr = sprintf('LGNc%d_SF%.1f_TF%d%s', LGNCtgr, lgnSF, lgnTF, ContrStr);
L6Str = sprintf('L6c%d_uplow%d-%d_%s', L6Ctgr, L6up, L6low, ExpTex);
candidates = { ...
    fullfile(DataFolder2, sprintf('Paper3_LIFLib_%s_Ang%s_%s_%s_%s.mat', DataPt, AngPrint, LGNStr, L6Str, DomFlag)), ...
    fullfile(DataFolder2, sprintf('Paper3_LIFLib_%s_Ang%.2f_%s_%s_%s.mat', DataPt, Angle, LGNStr, L6Str, DomFlag)), ...
    fullfile(DataFolder2, sprintf('Paper3_LIFLib_%s_Ang%.1f_%s_%s_%s.mat', DataPt, Angle, LGNStr, L6Str, DomFlag))};
for ii = 1:numel(candidates)
    if isfile(candidates{ii})
        FullPath = candidates{ii};
        return
    end
end
error('Missing full raw source for Angle=%.2f LGN=%d L6=%d ContrStr=%s. First tried: %s', Angle, LGNCtgr, L6Ctgr, ContrStr, candidates{1});
end
