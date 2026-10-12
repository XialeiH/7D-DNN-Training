%% LDE iteration w/o figures
% L4EmeshX,L4ImeshY,LDEFrfunc: compsed by several different response
% functions, and we choose the best from them for every step
%% Problem for the BG parts: maybe not fine enough! 
% Let's involve the previously computed local responses...
%% Starting from the finest function!
% Output: Iteration firing maps, and L1 diff norm

% New feature 051823: 
function [LDERepFinal,LDEIL2Diff,L2DiffNormNeib,L2Diameter,LDEequv,FuncUseAll,NANFinal] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6_noreorder(...
    PixLGNCtgr,L6Kernel,LDEIni,p,L6pars,Epoc,...
    C_SS_mean,C_CS_mean,C_IS_mean,...
    C_SC_mean,C_CC_mean,C_IC_mean,...
    C_SI_mean,C_CI_mean,C_II_mean,...
    L4SEp, L4SIp, ...
    L4CEp, L4CIp, ...
    L4IEp, L4IIp, ...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll, ...
    CtgrOrderReadout, alphaUse, ContrastUse, varargin)
NANFinal = false;

% specify NHCout
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
    EKp = varargin{6}; % excitation suppresion parameters
    IKp = varargin{7};
else
    %EKp.Thrsld = 50; EKp.Highist = [200,150]; EKp.HardBound = 200; EKp.Slope = 0; %%% NOT USED
    
    %IKp.Thrsld = 70; IKp.Highist = [100,95];  IKp.HardBound = 120; IKp.Slope = 0.95;
    IKp.Thrsld = 60; IKp.Highist = 120;  IKp.Slope = 0.9; % this is for new rule: multiplicative saturation
end

LDEItr = cell(Epoc+1,1);LDEItr{1} = LDEIni;
LDEEpoOut = cell(Epoc+1,1); LDEEpoOut{1} = LDEIni;
LDEoutVec = zeros(size(LDEIni.I,1)*3,Epoc+1);
LDEoutVec(:,1) = [LDEIni.S;LDEIni.C;LDEIni.I];

% record the function quoted every step
FuncUseAll = zeros(Epoc,1);

% Put all f(L4E, L4I, L6) into cells of 3D matrix at first
L6EMesh = 1:size(LDEFrfuncAll{1}.S,2);
L6MeshZ = L6EMesh';%reshape(L6EMesh, [1, 1, length(L6EMesh)]);
FuncN = length(L4EmeshXAll); % should be consistent for all three vars
lgnN = size(PixLGNCtgr,2);
FuncAll = cell(FuncN,1);
for funcInd = 1:FuncN
    LDEFrfunc = LDEFrfuncAll{funcInd};
    LDEFrfuncMatrix.S = zeros(lgnN,300,400,length(L6EMesh));
    LDEFrfuncMatrix.C = zeros(lgnN,300,400,length(L6EMesh));
    LDEFrfuncMatrix.I = zeros(lgnN,300,400,length(L6EMesh));
    for LGNInd = 1:lgnN
        for i = 1:length(L6EMesh)
            LDEFrfuncMatrix.S(LGNInd,:,:,i) = LDEFrfunc.S{LGNInd,i};
            LDEFrfuncMatrix.C(LGNInd,:,:,i) = LDEFrfunc.C{LGNInd,i};
            LDEFrfuncMatrix.I(LGNInd,:,:,i) = LDEFrfunc.I{LGNInd,i};
        end
    end
    FuncAll{funcInd} = LDEFrfuncMatrix;
end

NANGlobalFlag = false;
for Epc = 1:Epoc
    if NANGlobalFlag
        continue
    end
    
    LDEInpt = LDEItr{Epc};
    LDEUse = LDEInpt;
    if InhKillFlag % apply inhibition suppresion
        LDEUse.E = LDEUse.S * (1-0.3077) + LDEUse.C * 0.3077;
        LDEUseEAdj = L6Convert(LDEUse.E,EKp)./LDEUse.E;
        LDEUse.S = LDEUse.S .* LDEUseEAdj;
        LDEUse.C = LDEUse.C .* LDEUseEAdj;
        LDEUse.I   = InhMulp(LDEUse.I, IKp);
        %LDEUse.I   = InhKill(LDEUse.I, IKp.Thrsld, IKp.Highist, IKp.HardBound, IKp.Slope);
        %LDEUse.I   = InhKill(LDEUse.I, IKp.Thrsld, IKp.Highist, IKp.HardBound, IKp.Slope);%*0.8+LDEInpt.I*0.2 % compensate for the original recursive
    end
    LDEUse.E = LDEUse.S * (1-0.3077) + LDEUse.C * 0.3077;
    % Obtain the L4E/I inputs to every pixel's (S C I) neuron
    L4EUse_S = (C_SS_mean*LDEUse.S + C_SC_mean*LDEUse.C)/L4SEp;
    L4EUse_C = (C_CS_mean*LDEUse.S + C_CC_mean*LDEUse.C)/L4CEp;
    L4EUse_I = (C_IS_mean*LDEUse.S + C_IC_mean*LDEUse.C)/L4IEp;

    L4IUse_S = (C_SI_mean*LDEUse.I)/L4SIp;
    L4IUse_C = (C_CI_mean*LDEUse.I)/L4CIp;
    L4IUse_I = (C_II_mean*LDEUse.I)/L4IIp;
            
    % Obtain the L6 input
    FieldRow = N_HCOutY * NPixY;
    FieldCol = floor(length(LDEUse.E)/FieldRow);
    L4Efield = reshape(LDEUse.E,FieldRow,FieldCol); 
    L4Efield_padded = padarray(L4Efield, [1, 1], 'circular'); % Pad B for periodic boundary conditions    
    C = conv2(L4Efield_padded, L6Kernel, 'same'); % Perform convolution with periodic boundary conditions  
    %% The slope of L6 is adjustable
    L6EUse = L6Convert(C(2:end-1, 2:end-1),L6pars);% Extract the part of L6 that corresponds to the original dimensions of B
    
    %L61*(C(2:end-1, 2:end-1)-L62); 
    L6EUse(L6EUse>3*length(L6EMesh)) = 3*length(L6EMesh); L6EUse(L6EUse<3) = 3; % for now, I can't handel fL6>120 Hz since I didn't precompute that
    % compute the L6 plane that I should refer to. 3 = layer 1; 120 = layer
    % 40, and interpolate the rest.
    L6ELibInd = L6EUse(:)/3; % should be 1-33
    L6ELibInd(L6ELibInd<1) = 1; L6ELibInd(L6ELibInd>length(L6EMesh)) = length(L6EMesh); 
    % Choose the best option of domain: Large by defalt; if better move to
    % smalle and smaller
        
    % Refer to functions
    LDEOut = struct('S',[],'C',[],'I',[]);    
    % adjust for different functions
    nanFlag = true;
    FuncUse = 1; % using the finest by defalt, then go coarser if not good enough
    while nanFlag && FuncUse<=FuncN
        L4EmeshX = L4EmeshXAll{FuncUse};
        L4ImeshY = L4ImeshYAll{FuncUse};
        LDEFrfunc = FuncAll{FuncUse};
        LDEOut.S = LDEIterFunc_Grating_135Func_RealLGNL6_noreorder(...
            L4EmeshX, L4ImeshY,L6MeshZ,...
            LDEFrfunc.S,... % it's now a 5*33*300*400 mat, LGN, L6, L4I, L4E
            L4EUse_S,L4IUse_S,...
            PixLGNCtgr,L6ELibInd,CtgrOrderReadout,alphaUse,ContrastUse);
        %ORTPix,OBLPix,OPTPix,ORTOBLBd_Pix,OPTOBLBd_Pix);
        LDEOut.C = LDEIterFunc_Grating_135Func_RealLGNL6_noreorder(...
            L4EmeshX, L4ImeshY,L6MeshZ,...
            LDEFrfunc.C,...
            L4EUse_C,L4IUse_C,...
            PixLGNCtgr,L6ELibInd,CtgrOrderReadout,alphaUse,ContrastUse);
        % ORTPix,OBLPix,OPTPix,ORTOBLBd_Pix,OPTOBLBd_Pix);
        LDEOut.I = LDEIterFunc_Grating_135Func_RealLGNL6_noreorder(...
            L4EmeshX, L4ImeshY,L6MeshZ,...
            LDEFrfunc.I,...
            L4EUse_I,L4IUse_I,...
            PixLGNCtgr,L6ELibInd,CtgrOrderReadout,alphaUse,ContrastUse);
        %ORTPix,OBLPix,OPTPix,ORTOBLBd_Pix,OPTOBLBd_Pix);
        
        % Now check if output contains nan
        nanFlag = any(isnan([LDEOut.S,LDEOut.C,LDEOut.I]),'all');
        FuncUse = FuncUse+1;
    end
    
    
    % record the function used.
    FuncUseAll(Epc) = FuncUse-1;
    
    LDENext = struct(...
        'S',LDEOut.S*p + LDEInpt.S*(1-p),...
        'C',LDEOut.C*p + LDEInpt.C*(1-p),...
        'I',LDEOut.I*p + LDEInpt.I*(1-p));
    LDEItr{Epc+1} = LDENext; LDEEpoOut{Epc+1} = LDEOut;
    LDEoutVec(:,Epc+1) = [LDEOut.S; LDEOut.C; LDEOut.I];
    %BGFlagall(:,Epc+1) = [BGflagS;  BGflagC;  BGflagI];

    if FuncUse>FuncN && nanFlag % return if used up all functions but still getting nans
        fprintf('***Warning! NAN results in the %d epoch. Returning...\n',Epc)
        NANGlobalFlag = true;
    end
end

if strcmpi(Outflag,'f(xn)')
   %disp('Showing f(xn)')
   LDERepFinal = LDEoutVec;
elseif strcmpi(Outflag,'xn')
   %disp('Showing xn')
   LDERepFinal = LDEItr;
else
   disp('Unknown export flag, using xn')
   LDERepFinal = LDEItr;
end

%L2DiffNorm = zeros(Epoc,1);
L2DiffNormNeib = zeros(Epoc,1);
L2Diameter = zeros(Epoc,1);
LDEequv = [];
LDEIL2Diff = [];
% get an "equilibrium" of I, and assume >150 is fine
NANFinal = NANGlobalFlag;
if ~NANGlobalFlag
    LDEequv = mean(LDEoutVec(:,floor((Epc+1)*2/3):end),2);
    LDEIL2Diff = sqrt(sum((LDEoutVec - repmat(LDEequv,1,Epoc+1)).^2, 1));
    for Epc = 1:Epoc
        %     L2DiffNorm(Epc) = ...
        %         norm([LDEEpoOut{Epc}.S;LDEEpoOut{Epc}.C;LDEEpoOut{Epc}.I] - ...
        %         [LDEEpoOut{end}.S;LDEEpoOut{end}.C;LDEEpoOut{end}.I], 2);
        
        L2DiffNormNeib(Epc) = ...
            norm([LDEEpoOut{Epc}.S;LDEEpoOut{Epc}.C;LDEEpoOut{Epc}.I] - ...
            [LDEEpoOut{Epc+1}.S;LDEEpoOut{Epc+1}.C;LDEEpoOut{Epc+1}.I], 2);
        L2Diameter(Epc) = max(LDEIL2Diff(Epc:end));
    end
end
end
