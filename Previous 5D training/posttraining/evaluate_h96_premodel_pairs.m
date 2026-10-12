function evaluate_h96_premodel_pairs()
% Evaluate Paper2 SNN pixel inputs with the corrected h96 NN surrogate.

runtimeDir = getenv('H96_CORRECTED_RUNTIME');
if isempty(runtimeDir)
    runtimeDir = ['/scratch/xh2906/librarySCI_runs/' ...
        'sixd_prefAngle_varaware_singlemlp_h96_final70_smooth2d_low005_stratmid_torch_20260518_044134/' ...
        'hpc_runtime_bundle_extended_corrected_h96_ikp7_l6p5_matchedlib_20260702_010052'];
end
utilsDir = getenv('H96_CURRENT_UTILS');
simulationDir = getenv('SNN_OUTPUT_DIR');
outputFile = getenv('PREMODEL_PAIR_DATA');
if isempty(utilsDir) || isempty(simulationDir) || isempty(outputFile)
    error('H96_CURRENT_UTILS, SNN_OUTPUT_DIR, and PREMODEL_PAIR_DATA are required.');
end
addpath(utilsDir, '-begin');
addpath(runtimeDir, '-begin');

expectedHash = '41c7fdc837617cd66ee81b0a7f7dc2bc3af45ed8c38e67bc3e85abed8b6bb161';
resolvedResponse = which('LocalResponse_6D_MLP_prefAngle');
assert(startsWith(resolvedResponse, runtimeDir), ...
    'Wrong h96 LocalResponse resolved: %s', resolvedResponse);
assertSha256(resolvedResponse, expectedHash);
assertSha256(which('predict_pref6D_S'), ...
    '0eaf440aebc1ad3194c01efb3a807ab8eb03d8dff223a64fdd1565935da6049a');
assertSha256(which('predict_pref6D_C'), ...
    '1efcf8260cae702885f775e8875e098a9658d09aad7f74631ff701ec28709ae2');
assertSha256(which('predict_pref6D_I'), ...
    '6be47fb16e2b913f6cb7957b6b9daad9b6862f9b85fdd05822c4b54c9ac7c971');
assertSha256(which('L6Convert'), ...
    '5a08083283b921ea8bde6988ef3c8154a78f607be092dd385da94e861c4820e7');
assertSha256(which('L6ConvertRawLocal'), ...
    '444570cc9c9c125940fefaf44de9e63fcbf1baec5f512cd885749def01bbaa27');
assertSha256(which('InhMulp'), ...
    '6c74f749974b5199b3c3f76d6cf46fae1923f56f01e3d521398b4d26d421c68a');
fprintf('Corrected h96 runtime: %s\n', resolvedResponse);
fprintf('L6Convert: %s\n', which('L6Convert'));
fprintf('InhMulp: %s\n', which('InhMulp'));

L6ParamRawUse = {2.5,3,62.5,86,80.4,106,90,142.2,102,166.7,108,210,118, ...
    {[0 15 29],[0,34.0],[28 32 50],[69 85]}};
L6ParamUse = {2.5,3,{'c1smooth',L6ParamRawUse,unique(0:0.25:120)}};
IKp7 = struct('Thrsld1',54,'Thrsld2',63,'Highist',103,'Slope',0.941, ...
    'down1',0.3,'IntaH',4.5,'IntaL',-2.5,'IntbL',4,'IntbH',-0.3, ...
    'Mode','multisigmoid','SmoothJoinHalfWidth',[1 0 1], ...
    'SmoothT2QuinticWidth',3);
EKpUse = {1,0,[50;70;100],[50;60.5;69],'quadratic'};
L6Kernel = makeL6Kernel(0.75);
wC = 0.3077;
angles = [0, 7.5, 15, 22.5];
baseReplicate = 1;
extraReplicate = 2;
extraCountPerAngle = 270;
evenIndices = unique(round(linspace(1, 900, extraCountPerAngle)));
assert(numel(evenIndices) == extraCountPerAngle, 'Even 30%% sample has wrong size.');

E_premodel = []; E_h96 = []; I_premodel = []; I_h96 = [];
angle = []; pixelIndex = []; replicate = []; isExtra30 = [];
inputRanges = zeros(numel(angles) * 2, 6);
rangeRow = 0;
for ai = 1:numel(angles)
    for rep = [baseReplicate, extraReplicate]
        file = fullfile(simulationDir, sprintf( ...
            'NWSimulationPix_%.1fdeg_rep%02d_h96_validation.mat', angles(ai), rep));
        assert(isfile(file), 'Missing SNN simulation: %s', file);
        D = load(file, 'NWSmlt', 'metadata');
        [networkE, predictedE, networkI, predictedI, l6Input] = ...
            evaluateOne(D.NWSmlt, angles(ai), wC, L6Kernel, L6ParamUse, EKpUse, IKp7);
        rangeRow = rangeRow + 1;
        inputRanges(rangeRow,:) = [min(D.NWSmlt.L4E), max(D.NWSmlt.L4E), ...
            min(D.NWSmlt.L4I), max(D.NWSmlt.L4I), min(l6Input), max(l6Input)];
        if rep == baseReplicate
            keep = (1:900)';
            extraFlag = false(900,1);
        else
            keep = evenIndices(:);
            extraFlag = true(numel(keep),1);
        end
        E_premodel = [E_premodel; networkE(keep)]; %#ok<AGROW>
        E_h96 = [E_h96; predictedE(keep)]; %#ok<AGROW>
        I_premodel = [I_premodel; networkI(keep)]; %#ok<AGROW>
        I_h96 = [I_h96; predictedI(keep)]; %#ok<AGROW>
        angle = [angle; repmat(angles(ai),numel(keep),1)]; %#ok<AGROW>
        pixelIndex = [pixelIndex; keep]; %#ok<AGROW>
        replicate = [replicate; repmat(rep,numel(keep),1)]; %#ok<AGROW>
        isExtra30 = [isExtra30; extraFlag]; %#ok<AGROW>
    end
end

assert(numel(E_premodel) == 4680, 'Expected 4680 points after adding 30%%.');
assert(all(isfinite([E_premodel;E_h96;I_premodel;I_h96])), 'Nonfinite paired data.');
paired = struct('E_premodel',E_premodel,'E_h96',E_h96, ...
    'I_premodel',I_premodel,'I_h96',I_h96,'angleDeg',angle, ...
    'pixelIndex',pixelIndex,'replicate',replicate,'isExtra30',isExtra30);
provenance = struct('runtimeDir',runtimeDir,'localResponse',resolvedResponse, ...
    'localResponseSha256',expectedHash,'utilsDir',utilsDir, ...
    'simulationDir',simulationDir,'anglesDeg',angles, ...
    'baseReplicate',baseReplicate,'extraReplicate',extraReplicate, ...
    'basePointCount',3600,'extraPointCount',1080,'extraFraction',0.30, ...
    'extraPixelIndices',evenIndices,'L6ParamUse',{L6ParamUse}, ...
    'IKp7',IKp7,'EKpUse',{EKpUse},'inputRanges',inputRanges);
save(outputFile, 'paired', 'provenance', '-v7.3');
fprintf('Saved %s with %d paired points per population.\n', outputFile, numel(E_premodel));
end

function [networkE,predictedE,networkI,predictedI,l6Input] = ...
    evaluateOne(NWSmlt, angleDeg, wC, kernel, l6Pars, eKp, iKp)
networkE = NWSmlt.FS * (1-wC) + NWSmlt.FC * wC;
networkI = NWSmlt.FI;
field = reshape(networkE, 30, 30);
padded = field([end,1:end,1], [end,1:end,1]);
smoothed = conv2(padded, kernel, 'same');
l6Input = L6Convert(smoothed(2:end-1,2:end-1), l6Pars) / 3;
l6Input = min(max(l6Input(:), 1), 40);

lgnWeights = squeeze(sum(double(NWSmlt.PixInptCtgrUse), 3));
assert(isequal(size(lgnWeights), [900,4]), 'Unexpected Paper2 LGN category shape.');
rowSums = sum(lgnWeights,2);
assert(max(abs(rowSums-1)) < 1e-5, 'Paper2 LGN weights do not sum to one.');
pixLGN = [lgnWeights, zeros(900,1)];
alpha = angleDeg * ones(900,1);
contrast = 100 * ones(900,1);
rawS = evalPopulation('S',alpha,pixLGN,NWSmlt.L4E,NWSmlt.L4I,l6Input,contrast);
rawC = evalPopulation('C',alpha,pixLGN,NWSmlt.L4E,NWSmlt.L4I,l6Input,contrast);
rawI = evalPopulation('I',alpha,pixLGN,NWSmlt.L4E,NWSmlt.L4I,l6Input,contrast);
predictedE = L6Convert(rawS*(1-wC) + rawC*wC, eKp);
predictedI = InhMulp(rawI, iKp);
end

function result = evalPopulation(cellType,alpha,pixLGN,l4E,l4I,l6,contrast)
allCategories = zeros(numel(alpha),5);
for category = 1:5
    allCategories(:,category) = LocalResponse_6D_MLP_prefAngle(cellType,alpha, ...
        category*ones(size(alpha)),l4E,l4I,l6,contrast);
end
result = sum(pixLGN .* allCategories, 2);
end

function K = makeL6Kernel(sig)
trunc = 1.5 / sig;
kernelSize = max(2, floor(2 * sig * trunc));
[x,y] = meshgrid(1:kernelSize,1:kernelSize);
center = kernelSize / 2;
exponent = ((x-center).^2 + (y-center).^2) / (2*sig^2);
K = exp(-exponent);
K(K < exp(-(trunc^2)/2)) = 0;
K = K + K(end:-1:1,:) + K(:,end:-1:1) + K(end:-1:1,end:-1:1);
K = K / sum(K,'all');
K = K / ((sum(K,'all') - K(2,2)) / (1-0.3));
K(2,2) = 0.3;
end

function assertSha256(file, expected)
assert(~isempty(file) && isfile(file), 'Missing required MATLAB helper: %s', file);
[status, hashText] = system(sprintf('sha256sum "%s"', file));
assert(status == 0 && startsWith(strtrim(hashText), expected), ...
    'SHA256 mismatch for %s: %s', file, strtrim(hashText));
end
