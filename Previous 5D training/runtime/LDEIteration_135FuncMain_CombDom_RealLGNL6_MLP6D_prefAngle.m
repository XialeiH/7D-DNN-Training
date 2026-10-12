% 6D pref-angle MLP variant with Contrast + alpha inputs.

function [LDERepFinal,LDEIL2Diff,L2DiffNormNeib,L2Diameter,LDEequv,FuncUseAll,NANFinal] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle(...
    PixLGNCtgr,L6Kernel,LDEIni,p,L6pars,Epoc,...
    C_SS_mean,C_CS_mean,C_IS_mean,...
    C_SC_mean,C_CC_mean,C_IC_mean,...
    C_SI_mean,C_CI_mean,C_II_mean,...
    L4SEp, L4SIp, ...
    L4CEp, L4CIp, ...
    L4IEp, L4IIp, ...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll, ...
    ContrastUse, AlphaUse, varargin)

NANFinal = false;

if ~isequal(size(ContrastUse), size(AlphaUse), size(LDEIni.S))
    error('ContrastUse, AlphaUse, and LDE fields must share shape.');
end

if ~isempty(varargin)
    N_HCOutY = varargin{1};
    NPixX = varargin{2};
    NPixY = varargin{3};
else
    N_HCOutY = 4; NPixX = 10; NPixY = 10;
end

if length(varargin)>3
    InhKillFlag = varargin{4};
else
    InhKillFlag = true;
end

if length(varargin)>4
   Outflag = varargin{5};
else
   Outflag = 'xn';
end

if length(varargin)>5
    EKp = varargin{6};
    IKp = varargin{7};
else
    IKp.Thrsld = 60; IKp.Highist = 120; IKp.Slope = 0.9;
end

LDEItr = cell(Epoc+1,1);LDEItr{1} = LDEIni;
LDEEpoOut = cell(Epoc+1,1); LDEEpoOut{1} = LDEIni;
LDEoutVec = zeros(size(LDEIni.I,1)*3,Epoc+1);
LDEoutVec(:,1) = [LDEIni.S;LDEIni.C;LDEIni.I];

FuncUseAll = zeros(Epoc,1);

if ~isempty(LDEFrfuncAll) && isfield(LDEFrfuncAll{1}, 'S') && ~isempty(LDEFrfuncAll{1}.S)
    L6EMesh = 1:size(LDEFrfuncAll{1}.S,2);
else
    L6EMesh = 1:40;
end
FuncN = max(1, length(L4EmeshXAll));

NANGlobalFlag = false;
for Epc = 1:Epoc
    if NANGlobalFlag
        continue
    end

    LDEInpt = LDEItr{Epc};
    LDEUse = LDEInpt;
    if InhKillFlag
        LDEUse.E = LDEUse.S * (1-0.3077) + LDEUse.C * 0.3077;
        LDEUseEAdj = L6Convert(LDEUse.E,EKp)./LDEUse.E;
        LDEUse.S = LDEUse.S .* LDEUseEAdj;
        LDEUse.C = LDEUse.C .* LDEUseEAdj;
        LDEUse.I = InhMulp(LDEUse.I, IKp);
    end
    LDEUse.E = LDEUse.S * (1-0.3077) + LDEUse.C * 0.3077;

    L4EUse_S = (C_SS_mean*LDEUse.S + C_SC_mean*LDEUse.C)/L4SEp;
    L4EUse_C = (C_CS_mean*LDEUse.S + C_CC_mean*LDEUse.C)/L4CEp;
    L4EUse_I = (C_IS_mean*LDEUse.S + C_IC_mean*LDEUse.C)/L4IEp;

    L4IUse_S = (C_SI_mean*LDEUse.I)/L4SIp;
    L4IUse_C = (C_CI_mean*LDEUse.I)/L4CIp;
    L4IUse_I = (C_II_mean*LDEUse.I)/L4IIp;

    FieldRow = N_HCOutY * NPixY;
    FieldCol = floor(length(LDEUse.E)/FieldRow);
    L4Efield = reshape(LDEUse.E,FieldRow,FieldCol);
    L4Efield_padded = padarray(L4Efield, [1, 1], 'circular');
    C = conv2(L4Efield_padded, L6Kernel, 'same');
    L6EUse = L6Convert(C(2:end-1, 2:end-1),L6pars);

    L6EUse(L6EUse>3*length(L6EMesh)) = 3*length(L6EMesh); L6EUse(L6EUse<3) = 3;
    L6ELibInd = L6EUse(:)/3;
    L6ELibInd(L6ELibInd<1) = 1; L6ELibInd(L6ELibInd>length(L6EMesh)) = length(L6EMesh);

    LDEOut = struct('S',[],'C',[],'I',[]);
    nanFlag = true;
    FuncUse = 1;
    while nanFlag && FuncUse<=FuncN
        LDEOut.S = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
            'S',L4EUse_S,L4IUse_S,ContrastUse,AlphaUse,PixLGNCtgr,L6ELibInd);

        LDEOut.C = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
            'C',L4EUse_C,L4IUse_C,ContrastUse,AlphaUse,PixLGNCtgr,L6ELibInd);

        LDEOut.I = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
            'I',L4EUse_I,L4IUse_I,ContrastUse,AlphaUse,PixLGNCtgr,L6ELibInd);

        nanFlag = any(isnan([LDEOut.S,LDEOut.C,LDEOut.I]),'all');
        FuncUse = FuncUse+1;
    end

    FuncUseAll(Epc) = FuncUse-1;

    LDENext = struct(...
        'S',LDEOut.S*p + LDEInpt.S*(1-p),...
        'C',LDEOut.C*p + LDEInpt.C*(1-p),...
        'I',LDEOut.I*p + LDEInpt.I*(1-p));
    LDEItr{Epc+1} = LDENext; LDEEpoOut{Epc+1} = LDEOut;
    LDEoutVec(:,Epc+1) = [LDEOut.S; LDEOut.C; LDEOut.I];

    if FuncUse>FuncN && nanFlag
        fprintf('***Warning! NAN results in the %d epoch. Returning...\n',Epc)
        NANGlobalFlag = true;
    end
end

if strcmpi(Outflag,'f(xn)')
   LDERepFinal = LDEoutVec;
elseif strcmpi(Outflag,'xn')
   LDERepFinal = LDEItr;
else
   LDERepFinal = LDEItr;
end

L2DiffNormNeib = zeros(Epoc,1);
L2Diameter = zeros(Epoc,1);
LDEequv = [];
LDEIL2Diff = [];
NANFinal = NANGlobalFlag;
if ~NANGlobalFlag
    LDEequv = mean(LDEoutVec(:,floor((Epc+1)*2/3):end),2);
    LDEIL2Diff = sqrt(sum((LDEoutVec - repmat(LDEequv,1,Epoc+1)).^2, 1));
end

end
