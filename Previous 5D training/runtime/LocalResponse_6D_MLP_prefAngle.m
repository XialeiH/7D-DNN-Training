function LDEOutLIBy = LocalResponse_6D_MLP_prefAngle(celltype, alphaUse, LGNctgrUse, L4EUse, L4IUse, L6Use, ContrastUse) %#codegen
% Translate runtime variables into the 5D cos-only training contract:
% [L4E, L4I, L6, ContrastEff, OriCos2].
if ~isequal(size(alphaUse), size(LGNctgrUse), size(L4EUse), size(L4IUse), size(L6Use), size(ContrastUse))
    error('All LocalResponse inputs must share shape.');
end

thetaPref = zeros(size(alphaUse));
gammaUse = zeros(size(alphaUse));
oriCos2 = ones(size(alphaUse));
contrastEff = ContrastUse;

% High-L4E extrapolation guard: leave the trained support unchanged,
% then smoothly saturate L4E only above the training edge.
% Training support has L4E <= ~47500; the artifact branch uses ~66000.
L4EUse = h96_high_l4e_guard(L4EUse);

fgMask = (LGNctgrUse >= 1) & (LGNctgrUse <= 4);
if any(fgMask(:))
    thetaMap = [0, 135, 90, 45];
    thetaPref(fgMask) = thetaMap(double(LGNctgrUse(fgMask)));
    gammaUse(fgMask) = abs(wrapTo90_pref(double(alphaUse(fgMask)) - thetaPref(fgMask)));
    oriCos2(fgMask) = cosd(2 .* gammaUse(fgMask));
end

bgMask = (LGNctgrUse == 5);
if any(bgMask(:))
    contrastEff(bgMask) = 0;
    gammaUse(bgMask) = abs(wrapTo90_pref(double(alphaUse(bgMask)) - thetaPref(bgMask)));
    oriCos2(bgMask) = cosd(2 .* gammaUse(bgMask));
end

switch celltype
    case 'S'
        LDEOutLIBy = predict_pref6D_S(L4EUse, L4IUse, L6Use, contrastEff, oriCos2);
    case 'C'
        LDEOutLIBy = predict_pref6D_C(L4EUse, L4IUse, L6Use, contrastEff, oriCos2);
    case 'I'
        LDEOutLIBy = predict_pref6D_I(L4EUse, L4IUse, L6Use, contrastEff, oriCos2);
    otherwise
        error('Unknown celltype');
end
end

function thetaWrapped = wrapTo90_pref(thetaDeg)
thetaWrapped = mod(thetaDeg + 90, 180) - 90;
end


function xg = h96_high_l4e_guard(x)
edge = 47500;
full = 55000;
tailSlope = 0.02;
u = (double(x) - edge) ./ (full - edge);
u = min(max(u, 0), 1);
g = u.^3 .* (10 - 15*u + 6*u.^2);
xSat = edge + tailSlope .* (double(x) - edge);
xg = (1 - g) .* double(x) + g .* xSat;
xg = cast(xg, 'like', x);
end
