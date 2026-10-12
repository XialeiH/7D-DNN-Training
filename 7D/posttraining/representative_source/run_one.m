function run_one(repRoot,repIndex)
repCases=jsondecode(fileread(fullfile(repRoot,'cases.json')));repCase=repCases(repIndex+1);
AngleListOverride={sprintf('%.2f',repCase.angle)};ContrListOverride=repCase.contrast;
lgnTFOverride=repCase.tf;lgnSFOverride=repCase.sf;
RuntimeBundleDirOverride=fullfile(repRoot,'runtime');
SaveResultOverride=fullfile(repRoot,'data',sprintf('case_%03d.mat',repIndex));
if ~isfile(SaveResultOverride)
 run(fullfile(repRoot,'code/dense_nn_driver.m'));
end
repD=load(SaveResultOverride);
assert(all(isfinite([repD.S_map_full(:);repD.C_map_full(:);repD.I_map_full(:)])));
plot_source_pdf(fullfile(repRoot,'pdf'),cat(3,repD.S_map_full,repD.C_map_full,repD.I_map_full),[repCase.angle repCase.contrast repCase.tf repCase.sf],repD.final_step_max);
end
