function verify_runtime(root)
root=char(java.io.File(root).getCanonicalPath());
runtime=fullfile(root,'atlas_139','runtime_h96iter');addpath(runtime,'-begin');check_h96_library_mapping(runtime);
z=load(fullfile(root,'parity_inputs.mat'));X=z.X;names={'S','C','I'};baseline=[2.5 4.5 12.5];errors=zeros(1,3);
for k=1:3
 f=str2func(['predict_pref6D_' names{k}]);y=f(X(:,4),X(:,5),X(:,3),X(:,2),X(:,1),X(:,6),X(:,7));expected=z.(names{k});errors(k)=max(abs(y(:)-expected(:)));assert(all(isfinite(y))&&errors(k)<2e-4);
 bg=f([1000;2000;3000;4000;5000],1500,20,[0;14;14;14;14],0,[10;0;10;51;10],[2.5;2.5;0;2.5;16]);assert(all(bg==baseline(k)));
end
result=struct('points',64,'includes_TF32',true,'max_abs_errors',errors,'tolerance',2e-4,'baseline_triggers_verified',5,'mapping_verified',true);
fid=fopen(fullfile(root,'runtime_verification.json'),'w');fprintf(fid,'%s',jsonencode(result));fclose(fid);
end
