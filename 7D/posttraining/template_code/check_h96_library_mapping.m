function check_h96_library_mapping(libraryDir)
% Regression: foreground geometry is independent of contrast; BG stays fifth.
previousPath=path; restorePath=onCleanup(@() path(previousPath));
addpath(libraryDir,'-begin');
assert(strcmp(which('LDEIterFunc_Grating_135Func_RealLGNL6'), ...
    fullfile(libraryDir,'LDEIterFunc_Grating_135Func_RealLGNL6.m')));
V=zeros(5,2,2,2);for k=1:5,V(k,:,:,:)=10*k;end
P=[eye(5); .1 .2 .3 .15 .25]; q=.5*ones(6,1);
for angle=0:22.5:180
    order=rot90([2 1;3 4],floor(angle/45));
    if mod(floor(angle/22.5),2),order=flip(order,1);end
    order=[order(1,2);order(1,1);order(2,1);order(2,2)];
    expected=P*[10*order;50];
    for contrast=[5 14 24 95]
        actual=LDEIterFunc_Grating_135Func_RealLGNL6( ...
            [0 1],[0 1],[0 1],V,q,q,P,q,order,angle,contrast);
        assert(max(abs(actual-expected))<1e-12, ...
            'Channel mapping regression at angle=%g contrast=%g',angle,contrast);
    end
end
disp('PASS: 36 angle/contrast cases, all foreground channels, background and weighted mixing');
end
