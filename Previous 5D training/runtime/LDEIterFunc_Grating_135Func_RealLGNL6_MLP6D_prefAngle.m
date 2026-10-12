function LDEOutS = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
    celltype, L4EUse, L4IUse, ContrastUse, AlphaUse, PixLGNCtgr, L6AxisUse)

if ~isequal(size(L4EUse), size(L4IUse), size(ContrastUse), size(AlphaUse))
    error('L4EUse, L4IUse, ContrastUse, AlphaUse must have the same shape.');
end
if size(PixLGNCtgr, 1) ~= numel(L4EUse) || size(PixLGNCtgr, 2) ~= 5
    error('PixLGNCtgr must be N-by-5 where N = numel(L4EUse).');
end
LibyAll = zeros(length(L4EUse),5);
for LGNInd = 1:5
    LGNctgrUse = LGNInd * ones(size(AlphaUse));
    LibyAll(:,LGNInd) = LocalResponse_6D_MLP_prefAngle(...
        celltype, AlphaUse, LGNctgrUse, L4EUse, L4IUse, L6AxisUse, ContrastUse);
end
LDEOutS = sum(PixLGNCtgr .* LibyAll, 2);
end
