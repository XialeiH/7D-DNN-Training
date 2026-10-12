function simulate_paper2_snn_pixels()
% Run one Paper-2 SNN condition and save compact pixelwise firing-rate data.

projectRoot = getenv('PAPER2_PROJECT_ROOT');
if isempty(projectRoot)
    projectRoot = '/scratch/xh2906/NYU-Vision-2Drive-main';
end
outputDir = getenv('SNN_OUTPUT_DIR');
if isempty(outputDir)
    outputDir = fullfile(projectRoot, 'Data', 'Paper2_NetworkTuning', ...
        'Fig1V4', 'Paper2NWSimulationData_h96_validation');
end

angleDeg = envNumber('SNN_ANGLE', 0);
replicate = round(envNumber('SNN_REPLICATE', 1));
durationMs = envNumber('SNN_DURATION_MS', 10000);
analysisMs = envNumber('SNN_ANALYSIS_MS', min(9000, durationMs));
if analysisMs <= 0 || analysisMs > durationMs
    error('SNN_ANALYSIS_MS must be in (0, SNN_DURATION_MS].');
end

seed = 910000 + round(100 * angleDeg) * 10 + replicate;
rng(seed, 'twister');
addpath(projectRoot, '-begin');
addpath(fullfile(projectRoot, 'Utils'), '-begin');
if ~exist(outputDir, 'dir'); mkdir(outputDir); end

dataRoot = fullfile(projectRoot, 'Data', 'Paper2_NetworkTuning');
workspaceFile = fullfile(dataRoot, 'Fig1V4', 'GlobalConvTestWS.mat');
connectionFile = fullfile(dataRoot, 'DriveWkSp_SCSepa_Cconst_Conn.mat');
assert(isfile(workspaceFile), 'Missing Paper2 workspace: %s', workspaceFile);
assert(isfile(connectionFile), 'Missing Paper2 connectivity: %s', connectionFile);

W = load(workspaceFile);
C = load(connectionFile, 'CMatAll', 'EcplxInd');
C_EE = C.CMatAll.C_EE_Fix_Bd;
C_EI = C.CMatAll.C_EI_Fix_Bd;
C_IE = C.CMatAll.C_IE_Fix_Bd;
C_II = C.CMatAll.C_II_Fix_Bd;
EcplxInd = logical(C.EcplxInd(:));

N_E = W.N_E; N_I = W.N_I; N_S = W.N_S; N_C = W.N_C;
N_HC = W.N_HC; NPixX = W.NPixX; NPixY = W.NPixY;
dt = 0.1;
TPar = 10;
stepsTotal = round(durationMs / dt);
if mod(stepsTotal, TPar) ~= 0
    error('SNN_DURATION_MS/dt must be divisible by %d.', TPar);
end
stepsPerSection = stepsTotal / TPar;

% Orientation-dependent LGN and L6 drives from the Paper2 simulation.
ODNum = 4;
gratingHC = angleDeg + [0, 45, 90, 135];
L6up = 60; L6low = 6;
phaseFR = (cosd(abs(mod(gratingHC, 180)) * 2) + 1) / 2;
phaseFRS = [45 + phaseFR * 45, 45 - phaseFR * 45] / 1e3;
L6OrdF = (phaseFR * (L6up - L6low) + L6low) / 1e3;

LGNFilt = SpatialGaussianFilt_my(W.OD_SMap, N_HC, W.n_S_HC, ...
    W.n_S_HC * 0.2, 1.0, false);
LGNSon = LGNFilt * phaseFRS(1:4)' * W.N_Slgn;
LGNSoff = LGNFilt * phaseFRS(5:8)' * W.N_Slgn;
phaseFRC = ones(1, 8) * W.N_Clgn * (45 / 1e3);
LGNE = [[LGNSon; phaseFRC(W.OD_C)'], [LGNSoff; phaseFRC(W.OD_C + ODNum)']];

L6FiltS = SpatialGaussianFilt_my(W.OD_SMap, N_HC, W.n_S_HC, ...
    W.n_S_HC * 0.34, 1.25, false);
L6FiltC = SpatialGaussianFilt_my(W.OD_CMap, N_HC, W.n_C_HC, ...
    W.n_C_HC * 0.34, 1.25, false);
L6FiltI = SpatialGaussianFilt_my(W.OD_IMap, N_HC, W.n_I_HC, ...
    W.n_I_HC * 0.34, 1.25, false);
rE_L6 = [L6FiltS * L6OrdF' * W.NS_L6; L6FiltC * L6OrdF' * W.NC_L6];
rI_L6 = L6FiltI * L6OrdF' * W.NI_L6;

lambdaEPre = 45 * W.N_Slgn / 1e3;
lambdaIPre = 45 * W.N_Ilgn / 1e3;
inputPoolMs = durationMs * 0.05;
lgnEEvents = PoissonInputForNetwork(N_E, lambdaEPre, inputPoolMs, dt);
lgnIEvents = PoissonInputForNetwork(N_I, lambdaIPre, inputPoolMs, dt);
ambEEvents = PoissonInputForNetwork(N_E, W.rE_amb, inputPoolMs, dt);
ambIEvents = PoissonInputForNetwork(N_I, W.rI_amb, inputPoolMs, dt);
l6EEvents = PoissonInputForNetwork(N_E, rE_L6, inputPoolMs, dt);
l6IEvents = PoissonInputForNetwork(N_I, rI_L6, inputPoolMs, dt);
maxInputBin = floor(inputPoolMs / dt);

% Use the compatible saved Paper2 state, matching the source initialization.
state = W.EndState;
RefTimeE = state{1}; VE = state{2}; SpE = state{3};
GE_ampa_R = state{4}; GE_nmda_R = state{5}; GE_gaba_R = state{6};
GE_ampa_D = state{7}; GE_nmda_D = state{8}; GE_gaba_D = state{9};
RefTimeI = state{10}; VI = state{11}; SpI = state{12};
GI_ampa_R = state{13}; GI_nmda_R = state{14}; GI_gaba_R = state{15};
GI_ampa_D = state{16}; GI_nmda_D = state{17}; GI_gaba_D = state{18};
EEDlyRcd = sparse(N_E, 1);
IEDlyRcd = ones(N_I, 1);
EIDlyRcd = ones(N_E, 1);

LGNFreq = 4;
tMod = 1e3 / LGNFreq;
maxOnPhase = tMod / 2;
PhaseE = single(tMod * rand(N_E, 1));
frameMs = 100;
frameCount = round(frameMs / dt);
tt = 0:dt:(frameMs - dt);
spikeCountE = zeros(N_E, 1);
spikeCountI = zeros(N_I, 1);
analysisStartMs = durationMs - analysisMs;
timerAll = tic;

for section = 1:TPar
    timerSection = tic;
    for timeN = 1:stepsPerSection
        PhaseE = mod(PhaseE + dt, tMod);
        frameIndex = mod(timeN, frameCount);
        if frameIndex == 0; frameIndex = frameCount; end
        if frameIndex == 1
            randomBins = randi([1, maxInputBin + 1], 6, frameCount);
            phaseAll = floor(mod(PhaseE + tt, tMod) / maxOnPhase) * N_E + ...
                repmat((1:N_E)', 1, frameCount);
            adjustedLGNE = LGNE(phaseAll);
            EampaInp = single(full( ...
                W.S_Elgn * (lgnEEvents(:, randomBins(1,:)) .* adjustedLGNE / lambdaEPre) + ...
                W.S_amb * ambEEvents(:, randomBins(3,:)) + ...
                W.S_EL6 * l6EEvents(:, randomBins(5,:)) * W.rhoE_ampa));
            IampaInp = single(full( ...
                W.S_Ilgn * lgnIEvents(:, randomBins(2,:)) + ...
                W.S_amb * ambIEvents(:, randomBins(4,:)) + ...
                W.S_IL6 * l6IEvents(:, randomBins(6,:)) * W.rhoI_ampa));
            EnmdaInp = single(full(W.S_EL6 * l6EEvents(:, randomBins(5,:)) * W.rhoE_nmda));
            InmdaInp = single(full(W.S_IL6 * l6IEvents(:, randomBins(6,:)) * W.rhoI_nmda));
        end

        [RefTimeE,VE,SpE,GE_ampa_R,GE_nmda_R,GE_gaba_R, ...
            GE_ampa_D,GE_nmda_D,GE_gaba_D, ...
            RefTimeI,VI,SpI,GI_ampa_R,GI_nmda_R,GI_gaba_R, ...
            GI_ampa_D,GI_nmda_D,GI_gaba_D, ...
            EEDlyRcd,IEDlyRcd,EIDlyRcd] = ...
            V1NetworkUpdate_Ver3_Drive_PoissonRead_EEnIEnEIDelay( ...
            RefTimeE,VE,SpE,GE_ampa_R,GE_nmda_R,GE_gaba_R, ...
            GE_ampa_D,GE_nmda_D,GE_gaba_D, ...
            RefTimeI,VI,SpI,GI_ampa_R,GI_nmda_R,GI_gaba_R, ...
            GI_ampa_D,GI_nmda_D,GI_gaba_D, ...
            EEDlyRcd,IEDlyRcd,EIDlyRcd, C_EE,C_EI,C_IE,C_II, ...
            W.S_EE,W.S_EI,W.S_IE,W.S_II, ...
            W.tau_ampa_R,W.tau_ampa_D,W.tau_nmda_R,W.tau_nmda_D, ...
            W.tau_gaba_R,W.tau_gaba_D,W.tau_ref,dt,W.p_EEFail, ...
            W.gL_E,W.Ve,W.rhoE_ampa,W.rhoE_nmda, ...
            W.gL_I,W.Vi,W.rhoI_ampa,W.rhoI_nmda, ...
            EampaInp(:,frameIndex),IampaInp(:,frameIndex), ...
            EnmdaInp(:,frameIndex),InmdaInp(:,frameIndex));

        globalTimeMs = ((section - 1) * stepsPerSection + timeN) * dt;
        if globalTimeMs >= analysisStartMs
            idxE = find(SpE); idxI = find(SpI);
            spikeCountE(idxE) = spikeCountE(idxE) + 1;
            spikeCountI(idxI) = spikeCountI(idxI) + 1;
        end
        if sum(isnan(VE)) > 0.8 * N_E
            error('Paper2 SNN became nonfinite at %.3f ms.', globalTimeMs);
        end
    end
    fprintf('angle %.2f replicate %d section %d/%d: %.1f s\n', ...
        angleDeg, replicate, section, TPar, toc(timerSection));
end

FrE = spikeCountE * (1000 / analysisMs);
FrI = spikeCountI * (1000 / analysisMs);
[FrSMap,NnSPixel] = NeuVec2Pixel(FrE(~EcplxInd), W.NnS, NPixX*N_HC, NPixY*N_HC);
[FrCMap,~] = NeuVec2Pixel(FrE(EcplxInd), W.NnC, NPixX*N_HC, NPixY*N_HC);
[FrIMap,~] = NeuVec2Pixel(FrI, W.NnI, NPixX*N_HC, NPixY*N_HC);

L4EE = C_EE * FrE; L4EI = C_EI * FrI;
L4IE = C_IE * FrE; L4II = C_II * FrI;
[L4SE,~] = NeuVec2Pixel(L4EE(~EcplxInd), W.NnS, NPixX*N_HC, NPixY*N_HC);
[L4CE,~] = NeuVec2Pixel(L4EE(EcplxInd), W.NnC, NPixX*N_HC, NPixY*N_HC);
[L4IEMap,~] = NeuVec2Pixel(L4IE, W.NnI, NPixX*N_HC, NPixY*N_HC);
[L4SI,~] = NeuVec2Pixel(L4EI(~EcplxInd), W.NnS, NPixX*N_HC, NPixY*N_HC);
[L4CI,~] = NeuVec2Pixel(L4EI(EcplxInd), W.NnC, NPixX*N_HC, NPixY*N_HC);
[L4IIMap,~] = NeuVec2Pixel(L4II, W.NnI, NPixX*N_HC, NPixY*N_HC);

PixL6Ctgr = LGNIndSpat(L6FiltS, 1:4, NnSPixel, N_HC, N_HC, NPixX, NPixY);
PixLGNCtgr = LGNIndSpat(LGNFilt, 1:4, NnSPixel, N_HC, N_HC, NPixX, NPixY);
pixCount = N_HC^2 * NPixX * NPixY;
PixInptCtgrUse = zeros(pixCount, 4, 4, 'single');
for pixel = 1:pixCount
    PixInptCtgrUse(pixel,:,:) = PixLGNCtgr(pixel,:)' * PixL6Ctgr(pixel,:);
end

NWSmlt = struct();
NWSmlt.FS = FrSMap(:); NWSmlt.FC = FrCMap(:); NWSmlt.FI = FrIMap(:);
NWSmlt.L4E = (L4SE + L4CE + L4IEMap); NWSmlt.L4E = NWSmlt.L4E(:);
NWSmlt.L4I = (L4SI + L4CI + L4IIMap); NWSmlt.L4I = NWSmlt.L4I(:);
NWSmlt.PixInptCtgrUse = PixInptCtgrUse;

metadata = struct('angleDeg',angleDeg,'replicate',replicate,'seed',seed, ...
    'durationMs',durationMs,'analysisMs',analysisMs,'dtMs',dt, ...
    'runtimeSeconds',toc(timerAll),'networkE',N_E,'networkI',N_I, ...
    'connectionFile',connectionFile,'workspaceFile',workspaceFile);
outputFile = fullfile(outputDir, sprintf( ...
    'NWSimulationPix_%.1fdeg_rep%02d_h96_validation.mat', angleDeg, replicate));
save(outputFile, 'NWSmlt', 'metadata', '-v7.3');
fprintf('Saved %s\n', outputFile);
end

function value = envNumber(name, defaultValue)
textValue = getenv(name);
if isempty(textValue)
    value = defaultValue;
else
    value = str2double(textValue);
    if ~isfinite(value); error('Invalid numeric environment variable %s.', name); end
end
end
