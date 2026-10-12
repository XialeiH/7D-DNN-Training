% h96_library_mapping: geometric_all_contrasts_v1
%% Iterations of LDE: Use precomputed function to determine output of a cell type
function LDEOutS = LDEIterFunc_Grating_135Func_RealLGNL6(...
    L4EmeshX, L4ImeshY,L6MeshZ,...
    LDEFrfunc_Subf,...
    L4EUse,L4IUse,...
    PixLGNCtgr,L6ELibInd,CtgrOrderReadout,AlphaUse,ContrastUse)

    LDEOutLIBy = cell(size(LDEFrfunc_Subf,1),1);
    LibyAll = zeros(length(L4EUse),size(LDEFrfunc_Subf,1));
    [XX,YY,ZZ] = meshgrid(unique(L4EmeshX), unique(L4ImeshY), L6MeshZ);
    for LGNInd = 1:size(LDEFrfunc_Subf,1)
        LDEOutLIBy{LGNInd} = ...
            interp3(XX,YY,ZZ,...
             squeeze(LDEFrfunc_Subf(LGNInd,:,:,:)),...
            L4EUse, L4IUse, L6ELibInd, 'linear');
        LibyAll(:,LGNInd) = LDEOutLIBy{LGNInd};
    end

    % All contrasts use the same geometric foreground-channel mapping.
    if numel(CtgrOrderReadout) ~= 4
        error('CtgrOrderReadout must contain 4 reordered foreground categories.');
    end
    LibyFG = LibyAll(:,1:4);
    LibyAll(:,1:4) = LibyFG(:, CtgrOrderReadout(:)');

    LDEOutS = sum(PixLGNCtgr.*LibyAll,2);
end
