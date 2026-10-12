%% Iterations of LDE: Use precomputed function to determine output of a cell type
function LDEOutS = LDEIterFunc_Grating_135Func_RealLGNL6_noreorder(...
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

    contrastScalar = local_scalar_input(ContrastUse, 'ContrastUse');
    angleScalar = local_scalar_input(AlphaUse, 'AlphaUse');

    %#ok<NASGU>
    % Library-only validation: intentionally preserve LGN category order exactly
    % as stored in the aggregate library file. No low-contrast NN alignment and
    % no CtgrOrderReadout remapping are applied here.

    LDEOutS = sum(PixLGNCtgr.*LibyAll,2);
end

function scalarValue = local_scalar_input(x, name)
    x = x(:);
    if isempty(x)
        error('%s must not be empty.', name);
    end
    scalarValue = double(x(1));
    if any(abs(double(x) - scalarValue) > 1e-9)
        error('%s must be scalar or spatially constant.', name);
    end
end

function thetaWrapped = local_wrap_to90(thetaDeg)
    thetaWrapped = mod(thetaDeg + 90, 180) - 90;
end
