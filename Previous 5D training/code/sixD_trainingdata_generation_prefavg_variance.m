% Generate merged 6D preferred-angle training data with variance fields.
%
% Source selection per condition:
%   1. Prefer DS10_average when present; it carries LDEFrfuncVar.
%   2. Otherwise use normal DS10 when present.
%   3. Otherwise use DR100 as fallback.
%
% LGN5 is treated as background for every condition:
%   Contrast = 0, while Gamma/OriCos2/OriSin2 keep the stimulus-angle rule.
%
% Extra output fields:
%   FR_ResponseVar: across-repeat sample variance, or 0 if unavailable.
%   FR_ResponseN  : repeat count estimate, or 1 if unavailable.

clear; clc;

scriptDir  = fileparts(mfilename('fullpath'));
libraryDir = '/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/Paper3PlotingData';
stamp      = getenv('DATASET_STAMP');
if isempty(stamp)
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
end
saveDir = fullfile(scriptDir, ['prefavg_variance_DS10_21contr_' stamp]);
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

DataPt  = 'V4D2';
Comment = 'L6RealCtrlL4';
lgnSF   = 2.5;
lgnTF   = 10;
DomName = 'Small';

allowedAngles = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5];
allowedContr = [2 5 10 14 20 24 30 33 38 42 47 52 57 61 65 71 76 80 85 89 95];

lgnList   = 1:5;
l6List    = 1:40;
respTypes = {'S', 'C', 'I'};
targetOnly = strtrim(getenv('TARGET_ONLY'));
if ~isempty(targetOnly)
    if ~ismember(targetOnly, respTypes)
        error('TARGET_ONLY must be one of S, C, I; got %s', targetOnly);
    end
    respTypes = {targetOnly};
end
prefAngles = [0, 135, 90, 45];
outputSuffix = ['_prefavg_variance_DS10_21contr_' stamp];

fprintf('Output directory: %s\n', saveDir);
fprintf('Targets in this generation pass: %s\n', strjoin(respTypes, ','));

cases = struct('filePath', {}, 'fileName', {}, 'alpha', {}, 'contrast', {}, 'source', {}, 'hasVar', {});
caseIdx = 0;
for ai = 1:numel(allowedAngles)
    alphaVal = allowedAngles(ai);
    angleStr = sprintf('%.2f', alphaVal);
    for ci = 1:numel(allowedContr)
        contrVal = allowedContr(ci);
        [filePath, fileName, sourceTag, hasVar] = resolve_library_file( ...
            libraryDir, DataPt, Comment, angleStr, lgnSF, lgnTF, contrVal, DomName);
        if ~exist(filePath, 'file')
            error('Missing expected library file for angle %.2f contrast %d', alphaVal, contrVal);
        end
        caseIdx = caseIdx + 1;
        cases(caseIdx).filePath = filePath;
        cases(caseIdx).fileName = fileName;
        cases(caseIdx).alpha = alphaVal;
        cases(caseIdx).contrast = contrVal;
        cases(caseIdx).source = sourceTag;
        cases(caseIdx).hasVar = hasVar;
    end
end
fprintf('Validated %d expected condition files.\n', numel(cases));
fprintf('Average/variance-capable files: %d\n', sum([cases.hasVar]));

counts = zeros(numel(respTypes), 1);
caseRows = zeros(numel(cases), numel(respTypes));
negFilteredRows = zeros(numel(cases), numel(respTypes));
frMin = inf(numel(respTypes), 1);
frMax = -inf(numel(respTypes), 1);

fprintf('Prepass: counting rows and checking FR ranges...\n');
for f = 1:numel(cases)
    if f == 1 || mod(f, 10) == 0 || f == numel(cases)
        fprintf('[Prepass %d/%d] %s\n', f, numel(cases), cases(f).fileName);
    end
    lib = load(cases(f).filePath, 'LDEFrfunc', 'L4EmeshX', 'L4ImeshY');
    LDEFrfunc = lib.LDEFrfunc;
    L4I_vec = reshape(lib.L4ImeshY, [], 1);
    baseMask = (L4I_vec > 0);
    for r = 1:numel(respTypes)
        respName = respTypes{r};
        ncase = 0;
        for lgn = lgnList
            for l6 = l6List
                respGrid = reshape(LDEFrfunc.(respName){lgn, l6}, [], 1);
                if numel(respGrid) ~= numel(L4I_vec)
                    error('Size mismatch in %s resp=%s LGN=%d L6=%d', cases(f).fileName, respName, lgn, l6);
                end
                mask0 = baseMask & ~isnan(respGrid);
                vals0 = respGrid(mask0);
                negFilteredRows(f, r) = negFilteredRows(f, r) + sum(vals0 < 0);
                mask = mask0 & (respGrid >= 0);
                vals = respGrid(mask);
                ncase = ncase + sum(mask);
                if ~isempty(vals)
                    frMin(r) = min(frMin(r), min(vals));
                    frMax(r) = max(frMax(r), max(vals));
                end
            end
        end
        caseRows(f, r) = ncase;
        counts(r) = counts(r) + ncase;
    end
end

for r = 1:numel(respTypes)
    fprintf('Resp=%s total rows=%d FR min=%.6g max=%.6g\n', respTypes{r}, counts(r), frMin(r), frMax(r));
end

sourceRules = struct();
sourceRules.selectionPolicy = 'Prefer DS10_average if present, else DS10, else DR100 fallback';
sourceRules.allowedAngles = allowedAngles;
sourceRules.allowedContr = allowedContr;
sourceRules.libraryDir = libraryDir;
sourceRules.lgn5ContrastPolicy = 'LGNctgr 5 rows write Contrast=0 for every condition; angle encoding is preserved with thetaPref=0';
sourceRules.frFilterPolicy = 'Rows with that target FR_Response < 0 are removed independently for each target';
sourceRules.variancePolicy = 'FR_ResponseVar from LDEFrfuncVar when present; otherwise 0. FR_ResponseN from max repeat count when present; otherwise 1.';
sourceRules.caseFiles = {cases.fileName};
sourceRules.caseSources = {cases.source};
sourceRules.caseHasVar = [cases.hasVar];
sourceRules.caseRows = caseRows;
sourceRules.frMin = frMin;
sourceRules.frMax = frMax;

for r = 1:numel(respTypes)
    respName = respTypes{r};
    totalRows = counts(r);
    saveName = sprintf('%s_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part%s.mat', respName, outputSuffix);
    outPath = fullfile(saveDir, saveName);
    if exist(outPath, 'file')
        delete(outPath);
    end
    save(outPath, 'allowedAngles', 'allowedContr', 'prefAngles', 'sourceRules', '-v7.3');
    m = matfile(outPath, 'Writable', true);
    prealloc_dataset(m, totalRows);
end

writeAt = ones(numel(respTypes), 1);
fprintf('Writing datasets incrementally...\n');
for f = 1:numel(cases)
    if f == 1 || mod(f, 5) == 0 || f == numel(cases)
        fprintf('[Write %d/%d] %s\n', f, numel(cases), cases(f).fileName);
    end
    lib = load_library_with_optional_variance(cases(f).filePath);
    LDEFrfunc = lib.LDEFrfunc;
    LDEFrfuncVar = lib.LDEFrfuncVar;
    LDEFrRepeatCount = lib.LDEFrRepeatCount;
    L4E_vec = reshape(lib.L4EmeshX, [], 1);
    L4I_vec = reshape(lib.L4ImeshY, [], 1);
    baseMask = (L4I_vec > 0);
    alphaVal = cases(f).alpha;
    contrVal = cases(f).contrast;

    for r = 1:numel(respTypes)
        respName = respTypes{r};
        outPath = fullfile(saveDir, sprintf('%s_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part%s.mat', respName, outputSuffix));
        m = matfile(outPath, 'Writable', true);
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
                respGrid = reshape(LDEFrfunc.(respName){lgn, l6}, [], 1);
                varGrid = read_variance_grid(LDEFrfuncVar, respName, lgn, l6, numel(respGrid));
                repeatN = read_repeat_count_scalar(LDEFrRepeatCount, lgn, l6);
                if numel(varGrid) ~= numel(respGrid)
                    error('Variance size mismatch in %s resp=%s LGN=%d L6=%d', cases(f).fileName, respName, lgn, l6);
                end
                mask = baseMask & ~isnan(respGrid) & (respGrid >= 0);
                if ~any(mask)
                    continue;
                end
                vals = respGrid(mask);
                valsVar = varGrid(mask);
                valsVar(~isfinite(valsVar) | valsVar < 0) = 0;
                nmask = numel(vals);
                idx = writeAt(r):(writeAt(r) + nmask - 1);
                m.L4E(idx, 1) = single(L4E_vec(mask));
                m.L4I(idx, 1) = single(L4I_vec(mask));
                m.L6(idx, 1) = single(l6 * ones(nmask, 1));
                m.Contrast(idx, 1) = single(contrastEffVal * ones(nmask, 1));
                m.OriCos2(idx, 1) = single(oriCos2Val * ones(nmask, 1));
                m.OriSin2(idx, 1) = single(oriSin2Val * ones(nmask, 1));
                m.FR_Response(idx, 1) = single(vals);
                m.FR_ResponseVar(idx, 1) = single(valsVar);
                m.FR_ResponseN(idx, 1) = single(repeatN * ones(nmask, 1));
                m.LGNctgr(idx, 1) = int16(lgn * ones(nmask, 1));
                m.Alpha(idx, 1) = single(alphaVal * ones(nmask, 1));
                m.ThetaPref(idx, 1) = single(thetaPref * ones(nmask, 1));
                m.Gamma(idx, 1) = single(gammaVal * ones(nmask, 1));
                writeAt(r) = writeAt(r) + nmask;
            end
        end
    end
end

for r = 1:numel(respTypes)
    if writeAt(r) ~= counts(r) + 1
        error('Write count mismatch for %s: wrote %d expected %d', respTypes{r}, writeAt(r)-1, counts(r));
    end
end

if numel(respTypes) == 1
    manifestName = sprintf('generation_manifest_%s%s.mat', respTypes{1}, outputSuffix);
else
    manifestName = sprintf('generation_manifest%s.mat', outputSuffix);
end
manifestPath = fullfile(saveDir, manifestName);
save(manifestPath, 'allowedAngles', 'allowedContr', 'prefAngles', 'sourceRules', ...
    'counts', 'caseRows', 'negFilteredRows', 'frMin', 'frMax', 'respTypes', '-v7.3');
fprintf('Saved generation manifest: %s\n', manifestPath);
fprintf('All done.\n');

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

function prealloc_dataset(m, totalRows)
    m.L4E(totalRows, 1) = single(0);
    m.L4I(totalRows, 1) = single(0);
    m.L6(totalRows, 1) = single(0);
    m.Contrast(totalRows, 1) = single(0);
    m.OriCos2(totalRows, 1) = single(0);
    m.OriSin2(totalRows, 1) = single(0);
    m.FR_Response(totalRows, 1) = single(0);
    m.FR_ResponseVar(totalRows, 1) = single(0);
    m.FR_ResponseN(totalRows, 1) = single(1);
    m.LGNctgr(totalRows, 1) = int16(0);
    m.Alpha(totalRows, 1) = single(0);
    m.ThetaPref(totalRows, 1) = single(0);
    m.Gamma(totalRows, 1) = single(0);
end

function y = wrapTo90_signed(x)
    y = mod(x + 90, 180) - 90;
end
