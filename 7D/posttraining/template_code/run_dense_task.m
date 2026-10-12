function run_dense_task(root, runtime)
T = readtable(fullfile(root,'cases.tsv'),'FileType','text','Delimiter','\t');
k = str2double(getenv('SLURM_ARRAY_TASK_ID'))+1;
AngleListOverride = {sprintf('%.2f',T.angle(k))};
ContrListOverride = T.contrast(k);
lgnTFOverride = T.tf(k); lgnSFOverride = T.sf(k);
RuntimeBundleDirOverride = runtime;
SaveResultOverride = fullfile(root,'equilibrium',sprintf('case_%03d.mat',k-1));
if isfile(SaveResultOverride), return; end
run(fullfile(root,'code','dense_nn_driver.m'));
end
