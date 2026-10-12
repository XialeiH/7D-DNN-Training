function run_case()
root = '/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h512x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261002_original_loss/posttraining_7d_tf32_globalall7/atlas_139';
runtime = '/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h512x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261002_original_loss/posttraining_7d_tf32_globalall7/atlas_139/runtime_h96iter';
driver = fullfile(root, 'eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition.m');
casesPath = getenv('H96_ATLAS_CASES');
task = str2double(getenv('SLURM_ARRAY_TASK_ID'));
T = readtable(casesPath, 'FileType', 'text', 'Delimiter', '	', 'VariableNamingRule', 'preserve');
row = T(T.task == task, :);
if height(row) ~= 1, error('Expected one case for task %d', task); end
angle = row.angle(1);
contrast = row.contrast(1);
lgnTF = row.tf(1);
lgnSF = row.sf(1);
libDir = char(string(row.library_dir(1)));
saveDir = char(string(row.save_dir(1)));
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
addpath(runtime, '-begin');
cd('/scratch/xh2906/NYU-Vision-2Drive-main');
AngleListOverride = {sprintf('%.2f', angle)};
ContrListOverride = contrast;
SaveBaseDirOverride = saveDir;
SaveFigOverride = false;
SavePngOverride = true;
RuntimeBundleDirOverride = runtime;
BGFlagOverride = false;
LibCacheDirOverride = '__NO_CACHE__';
LowContrastLibCacheDirOverride = '__NO_CACHE__';
Extra50Case = row.case(1) > 102;
if Extra50Case
    LibCacheDirOverride = fullfile(fileparts(libDir), 'equilibrium_library');
    if ~isfolder(LibCacheDirOverride), error('Missing extra50 library cache'); end
end
SaveToFolderOverride = libDir;
lgnTFOverride = lgnTF;
lgnSFOverride = lgnSF;
global H96_LGN_TF_USE H96_LGN_SF_USE
H96_LGN_TF_USE = lgnTF;
H96_LGN_SF_USE = lgnSF;
run(driver);
end
