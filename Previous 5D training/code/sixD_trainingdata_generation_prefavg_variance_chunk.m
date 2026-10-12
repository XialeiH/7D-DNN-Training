% Generate one preferred-angle variance-aware training chunk for one target
% and one condition. This is used to avoid writing a giant v7.3 target MAT
% file serially. Each Slurm array task writes one independent chunk file.

clear; clc;

scriptDir  = fileparts(mfilename('fullpath'));
libraryDir = '/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/Paper3PlotingData';
stamp      = getenv('DATASET_STAMP');
targetOnly = strtrim(getenv('TARGET_ONLY'));
caseIndexRaw = strtrim(getenv('CASE_INDEX'));
if isempty(stamp)
    error('DATASET_STAMP is required');
end
if ~ismember(targetOnly, {'S', 'C', 'I'})
    error('TARGET_ONLY must be one of S, C, I; got %s', targetOnly);
end
if isempty(caseIndexRaw)
    error('CASE_INDEX is required');
end
caseIndex = str2double(caseIndexRaw);

saveDir = fullfile(scriptDir, ['prefavg_variance_DS10_21contr_' stamp]);
chunkDir = fullfile(saveDir, ['chunks_' targetOnly]);
if ~exist(chunkDir, 'dir')
    mkdir(chunkDir);
end

DataPt  = 'V4D2';
Comment = 'L6RealCtrlL4';
lgnSF   = 2.5;
lgnTF   = 10;
DomName = 'Small';

allowedAngles = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5];
allowedContr = [2 5 10 14 20 24 30 33 38 42 47 52 57 61 65 71 76 80 85 89 95];
prefAngles = [0, 135, 90, 45];
lgnList = 1:5;
l6List = 1:40;

nContr = numel(allowedContr);
nCases = numel(allowedAngles) * nContr;
if ~isfinite(caseIndex) || caseIndex < 1 || caseIndex > nCases || floor(caseIndex) ~= caseIndex
    error('CASE_INDEX must be integer in 1..%d; got %s', nCases, caseIndexRaw);
end
angleIdx = floor((caseIndex - 1) / nContr) + 1;
contrIdx = mod(caseIndex - 1, nContr) + 1;
alphaVal = allowedAngles(angleIdx);
contrVal = allowedContr(contrIdx);
angleStr = sprintf('%.2f', alphaVal);

[filePath, fileName, sourceTag, hasVar] = resolve_library_file( ...
    libraryDir, DataPt, Comment, angleStr, lgnSF, lgnTF, contrVal, DomName);
if ~exist(filePath, 'file')
    error('Missing library file for case %d angle %.2f contrast %d', caseIndex, alphaVal, contrVal);
end
fprintf('Target=%s case=%d/%d angle=%.2f contrast=%d source=%s file=%s\n', ...
    targetOnly, caseIndex, nCases, alphaVal, contrVal, sourceTag, fileName);

lib = load_library_with_optional_variance(filePath);
LDEFrfunc = lib.LDEFrfunc;
LDEFrfuncVar = lib.LDEFrfuncVar;
LDEFrRepeatCount = lib.LDEFrRepeatCount;
L4E_vec = reshape(lib.L4EmeshX, [], 1);
L4I_vec = reshape(lib.L4ImeshY, [], 1);
baseMask = (L4I_vec > 0);

totalRows = 0;
negFilteredRows = 0;
frMin = inf;
frMax = -inf;
for lgn = lgnList
    for l6 = l6List
        respGrid = reshape(LDEFrfunc.(targetOnly){lgn, l6}, [], 1);
        if numel(respGrid) ~= numel(L4I_vec)
            error('Size mismatch in %s resp=%s LGN=%d L6=%d', fileName, targetOnly, lgn, l6);
        end
        mask0 = baseMask & ~isnan(respGrid);
        vals0 = respGrid(mask0);
        negFilteredRows = negFilteredRows + sum(vals0 < 0);
        mask = mask0 & (respGrid >= 0);
        vals = respGrid(mask);
        totalRows = totalRows + sum(mask);
        if ~isempty(vals)
            frMin = min(frMin, min(vals));
            frMax = max(frMax, max(vals));
        end
    end
end

L4E = zeros(totalRows, 1, 'single');
L4I = zeros(totalRows, 1, 'single');
L6 = zeros(totalRows, 1, 'single');
Contrast = zeros(totalRows, 1, 'single');
OriCos2 = zeros(totalRows, 1, 'single');
OriSin2 = zeros(totalRows, 1, 'single');
FR_Response = zeros(totalRows, 1, 'single');
FR_ResponseVar = zeros(totalRows, 1, 'single');
FR_ResponseN = ones(totalRows, 1, 'single');
Gamma = zeros(totalRows, 1, 'single');
LGNctgr = zeros(totalRows, 1, 'int16');
Alpha = zeros(totalRows, 1, 'single');
ThetaPref = zeros(totalRows, 1, 'single');

writeAt = 1;
for lgn = lgnList
    if lgn >= 1 && lgn <= 4
        thetaPref = prefAngles(lgn);
        gammaVal = abs(wrapTo90_signed(alphaVal - thetaPref));
        oriCos2Val = cosd(2 * gammaVal);
        oriSin2Val = sind(2 * gammaVal);
        contrastEffVal = contrVal;
    elseif lgn == 5
        thetaPref = 0;
        gammaVal = abs(wrapTo90_signed(alphaVal - thetaPref));
        oriCos2Val = cosd(2 * gammaVal);
        oriSin2Val = sind(2 * gammaVal);
        contrastEffVal = 0;
    else
        error('Unexpected LGN index: %d', lgn);
    end
    for l6 = l6List
        respGrid = reshape(LDEFrfunc.(targetOnly){lgn, l6}, [], 1);
        varGrid = read_variance_grid(LDEFrfuncVar, targetOnly, lgn, l6, numel(respGrid));
        repeatN = read_repeat_count_scalar(LDEFrRepeatCount, lgn, l6);
        mask = baseMask & ~isnan(respGrid) & (respGrid >= 0);
        if ~any(mask)
            continue;
        end
        vals = respGrid(mask);
        valsVar = varGrid(mask);
        valsVar(~isfinite(valsVar) | valsVar < 0) = 0;
        nmask = numel(vals);
        idx = writeAt:(writeAt + nmask - 1);
        L4E(idx) = single(L4E_vec(mask));
        L4I(idx) = single(L4I_vec(mask));
        L6(idx) = single(l6);
        Contrast(idx) = single(contrastEffVal);
        OriCos2(idx) = single(oriCos2Val);
        OriSin2(idx) = single(oriSin2Val);
        FR_Response(idx) = single(vals);
        FR_ResponseVar(idx) = single(valsVar);
        FR_ResponseN(idx) = single(repeatN);
        Gamma(idx) = single(gammaVal);
        LGNctgr(idx) = int16(lgn);
        Alpha(idx) = single(alphaVal);
        ThetaPref(idx) = single(thetaPref);
        writeAt = writeAt + nmask;
    end
end
if writeAt ~= totalRows + 1
    error('Write count mismatch: wrote %d expected %d', writeAt - 1, totalRows);
end

chunkName = sprintf('%s_case%03d_Ang%s_Contr%d_%s.mat', targetOnly, caseIndex, angleStr, contrVal, sourceTag);
chunkPath = fullfile(chunkDir, chunkName);
if exist(chunkPath, 'file')
    delete(chunkPath);
end
save(chunkPath, 'L4E', 'L4I', 'L6', 'Contrast', 'OriCos2', 'OriSin2', ...
    'FR_Response', 'FR_ResponseVar', 'FR_ResponseN', 'Gamma', 'LGNctgr', ...
    'Alpha', 'ThetaPref', 'targetOnly', 'caseIndex', 'alphaVal', 'contrVal', ...
    'sourceTag', 'hasVar', 'fileName', 'totalRows', 'negFilteredRows', ...
    'frMin', 'frMax', '-v7.3');
fprintf('Saved chunk: %s rows=%d FR=[%.6g, %.6g] negFiltered=%d\n', ...
    chunkPath, totalRows, frMin, frMax, negFilteredRows);

function [filePath, fileName, sourceTag, hasVar] = resolve_library_file(libraryDir, DataPt, Comment, angleStr, lgnSF, lgnTF, contrVal, DomName)
    candidates = {'DS10_average', 'DS10', 'DR100'};
    hasVarByCandidate = [true, false, false];
    for ii = 1:numel(candidates)
        tag = candidates{ii};
        fileName = sprintf('Func200%s-%s-Ang%s-SF%.1fTF%d_Contr%d-%s_LGNstep3p75_%s_part.mat', ...
            DataPt, Comment, angleStr, lgnSF, lgnTF, contrVal, DomName, tag);
        filePath = fullfile(libraryDir, fileName);
        if exist(filePath, 'file')
            sourceTag = tag;
            hasVar = hasVarByCandidate(ii);
            return;
        end
    end
    filePath = '';
    fileName = '';
    sourceTag = '';
    hasVar = false;
end

function lib = load_library_with_optional_variance(filePath)
    info = whos('-file', filePath);
    names = {info.name};
    vars = {'LDEFrfunc', 'L4EmeshX', 'L4ImeshY'};
    hasVar = ismember('LDEFrfuncVar', names);
    hasRepeat = ismember('LDEFrRepeatCount', names);
    if hasVar
        vars{end+1} = 'LDEFrfuncVar';
    end
    if hasRepeat
        vars{end+1} = 'LDEFrRepeatCount';
    end
    lib = load(filePath, vars{:});
    if ~hasVar
        lib.LDEFrfuncVar = [];
    end
    if ~hasRepeat
        lib.LDEFrRepeatCount = [];
    end
end

function varGrid = read_variance_grid(LDEFrfuncVar, respName, lgn, l6, expectedN)
    if isempty(LDEFrfuncVar) || ~isfield(LDEFrfuncVar, respName)
        varGrid = zeros(expectedN, 1);
        return;
    end
    cellArr = LDEFrfuncVar.(respName);
    if size(cellArr, 1) < lgn || size(cellArr, 2) < l6 || isempty(cellArr{lgn, l6})
        varGrid = zeros(expectedN, 1);
        return;
    end
    varGrid = reshape(cellArr{lgn, l6}, [], 1);
end

function repeatN = read_repeat_count_scalar(LDEFrRepeatCount, lgn, l6)
    repeatN = 1;
    if isempty(LDEFrRepeatCount)
        return;
    end
    try
        if ndims(LDEFrRepeatCount) >= 4 && size(LDEFrRepeatCount, 3) >= lgn && size(LDEFrRepeatCount, 4) >= l6
            vals = double(LDEFrRepeatCount(:, :, lgn, l6));
        elseif ndims(LDEFrRepeatCount) >= 2 && size(LDEFrRepeatCount, 1) >= lgn && size(LDEFrRepeatCount, 2) >= l6
            vals = double(LDEFrRepeatCount{lgn, l6});
        else
            vals = double(LDEFrRepeatCount(:));
        end
        vals = vals(isfinite(vals) & vals > 0);
        if ~isempty(vals)
            repeatN = max(1, round(max(vals)));
        end
    catch
        repeatN = 1;
    end
end

function y = wrapTo90_signed(x)
    y = mod(x + 90, 180) - 90;
end
