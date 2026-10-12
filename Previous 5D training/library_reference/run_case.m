case_tsv = getenv('CASE_TSV');
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
T = readtable(case_tsv, 'FileType', 'text', 'Delimiter', '\t');
row = T(T.task == task_id,:);
if height(row) ~= 1
    error('Expected one case for task %d, got %d', task_id, height(row));
end
AngleListOverride = {sprintf('%.2f', row.angle(1))};
ContrListOverride = row.contrast(1);
SaveBaseDirOverride = getenv('SMROOT');
LibrarySourceDirOverride = getenv('LIBSRC');
RuntimeBundleDirOverride = getenv('SMROOT');
run(fullfile(getenv('SMROOT'), 'eigenvalue_6D_library_only_70_fullnods_smooth2d.m'));
