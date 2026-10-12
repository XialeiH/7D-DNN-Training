%% for multiplicative depressions of firing rates
% LDEUse: any vector - the operation is entrywise
% IKp: IKp.Thrsld = 60; IKp.Highist = 120;  IKp.Slope = 0.9; before thrsld,
% 
% the factor is 1, after 120 is 0.9, between them comes from linear
% interpolation

function [LDEUseOut,Factor]  = InhMulp(LDEUseI, IKp1)
Factor = ones(size(LDEUseI));
Factor(LDEUseI>=IKp1.Highist) = IKp1.Slope;

if strcmp(IKp1.Mode,'linear') == 1
    InterpId = LDEUseI<IKp1.Highist & LDEUseI>IKp1.Thrsld;
    Factor(InterpId) = 1 + (LDEUseI(InterpId) - IKp1.Thrsld)/(IKp1.Highist - IKp1.Thrsld) * (IKp1.Slope - 1);
elseif strcmp(IKp1.Mode,'sigmoid') == 1
    % using -6 to 6 to interpolate:
    InterpId = LDEUseI<IKp1.Highist & LDEUseI>IKp1.Thrsld;
    aa = IKp1.IntH; bb = IKp1.IntL;
    xx = (LDEUseI(InterpId) - IKp1.Thrsld)/(IKp1.Highist - IKp1.Thrsld) * (aa - -bb) -bb;
    sigm = sigmod(xx);
    high = sigmod(aa); low = sigmod(-bb);
    % rescale the outcome of sigm
    sigmRescle = (sigm - low)/(high-low);
    Factor(InterpId) = 1 + sigmRescle * (IKp1.Slope - 1);
elseif strcmp(IKp1.Mode,'multisigmoid') == 1 % multiple lags of sigmoid
    if isfield(IKp1,'down1')
        down1 = IKp1.down1; % how much does the first part goes down? should be between 0 and 1
    else
        down1 = 0.5; % 0.5 by default
    end
    down2 = 1-down1;
    InterpId2 = LDEUseI<IKp1.Highist & LDEUseI>=IKp1.Thrsld2;
    InterpId1 = LDEUseI<=IKp1.Thrsld2 & LDEUseI>=IKp1.Thrsld1;
    aL = IKp1.IntaL; bL = IKp1.IntbL;
    aH = IKp1.IntaH; bH = IKp1.IntbH;
    % for lag 1
    xxbb = (LDEUseI(InterpId1) - IKp1.Thrsld1)/(IKp1.Thrsld2 - IKp1.Thrsld1) * (bH+bL) -bL;
    sigmbb = sigmod(xxbb);
    highbb = sigmod(bH); lowbb = sigmod(-bL);
    % rescale the outcome of sigm
    sigmResclebb = (sigmbb - lowbb)/(highbb-lowbb);
    sigmResclebb = sigmResclebb * down1; % fixed endpoint normalization; avoids map-dependent kink at Thrsld2
    Factor(InterpId1) = 1 + sigmResclebb * (IKp1.Slope - 1);
    %% for lag 2
    highaa = sigmod(aH); lowaa = sigmod(-aL);

    xxaa = (LDEUseI(InterpId2) - IKp1.Thrsld2)/(IKp1.Highist - IKp1.Thrsld2) * (aH+aL) -aL;
    sigmaa = sigmod(xxaa);   
    % rescale the outcome of sigm
    sigmRescleaa = (sigmaa - lowaa)/(highaa-lowaa);
    
	    sigmRescleaa = sigmRescleaa * down2 + down1;
	    Factor(InterpId2) = 1 + sigmRescleaa * (IKp1.Slope - 1);
	    if isfield(IKp1,'SmoothJoinHalfWidth') && any(IKp1.SmoothJoinHalfWidth > 0)
	        Factor = smoothMultisigmoidJoins(LDEUseI, Factor, IKp1, down1, down2);
	    end
	    if isfield(IKp1,'SmoothT2QuinticWidth') && IKp1.SmoothT2QuinticWidth > 0
	        Factor = smoothT2Quintic(LDEUseI, Factor, IKp1, down1, down2);
	    end
	elseif strcmp(IKp1.Mode,'threesegments') == 1
    if isfield(IKp1,'down1')
        down1 = IKp1.down1; % how much does the first part goes down? should be between 0 and 1
    else
        down1 = 0.5; % 0.5 by default
    end
    down2 = 1-down1;
    if isfield(IKp1,'Thrsld3')
        Thrsld3 = IKp1.Thrsld3;
    else
        Thrsld3 = 65;
    end
    if isfield(IKp1,'MidCurve')
        MidCurve = IKp1.MidCurve;
    else
        MidCurve = 0;
    end
    TailIKp = frozenTailParams(IKp1);
    tailDown1 = TailIKp.down1;
    tailDown2 = 1-tailDown1;
    tailSlope = TailIKp.Slope;
    InterpId1 = LDEUseI<=IKp1.Thrsld2 & LDEUseI>=IKp1.Thrsld1;
    InterpIdMid = LDEUseI<Thrsld3 & LDEUseI>IKp1.Thrsld2;
    InterpId2 = LDEUseI<TailIKp.Highist & LDEUseI>=Thrsld3;
    aL = IKp1.IntaL; bL = IKp1.IntbL;
    aH = IKp1.IntaH; bH = IKp1.IntbH;

    xxbb = (LDEUseI(InterpId1) - IKp1.Thrsld1)/(IKp1.Thrsld2 - IKp1.Thrsld1) * (bH+bL) -bL;
    sigmbb = sigmod(xxbb);
    highbb = sigmod(bH); lowbb = sigmod(-bL);
    sigmResclebb = (sigmbb - lowbb)/(highbb-lowbb);
    sigmResclebb = sigmResclebb * down1;
    Factor(InterpId1) = 1 + sigmResclebb * (IKp1.Slope - 1);

    [tailEndProgress, tailEndSlope] = secondSegmentProgressAndSlope(Thrsld3, TailIKp, tailDown1, tailDown2);
    midStartFactor = 1 + down1 * (IKp1.Slope - 1);
    midEndFactor = 1 + tailEndProgress * (tailSlope - 1);
    firstEndSlope = down1 * highbb * (1-highbb) * (bH+bL) / ...
        ((IKp1.Thrsld2 - IKp1.Thrsld1) * (highbb-lowbb)) * (IKp1.Slope - 1);
    tailEndSlope = tailEndSlope * (tailSlope - 1);
    if any(InterpIdMid,'all')
        t = (LDEUseI(InterpIdMid) - IKp1.Thrsld2)/(Thrsld3 - IKp1.Thrsld2);
        deltaX = Thrsld3 - IKp1.Thrsld2;
        h00 = 2*t.^3 - 3*t.^2 + 1;
        h10 = t.^3 - 2*t.^2 + t;
        h01 = -2*t.^3 + 3*t.^2;
        h11 = t.^3 - t.^2;
        Factor(InterpIdMid) = h00*midStartFactor + h10*deltaX*firstEndSlope + ...
            h01*midEndFactor + h11*deltaX*tailEndSlope + ...
            MidCurve * (tailSlope - 1) * t.^2 .* (1-t).^2;
    end

	    if any(InterpId2,'all')
	        Factor(InterpId2) = 1 + secondSegmentProgressAndSlope(LDEUseI(InterpId2), TailIKp, tailDown1, tailDown2) * (tailSlope - 1);
	    end
		    if isfield(IKp1,'EarlyShiftPoints') && isfield(IKp1,'EarlyShiftAmp')
		        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.EarlyShiftPoints, IKp1.EarlyShiftAmp);
		    end
			    if isfield(IKp1,'LocalShiftPoints') && isfield(IKp1,'LocalShiftAmp')
			        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.LocalShiftPoints, IKp1.LocalShiftAmp);
			    end
			    if isfield(IKp1,'ExtraLocalShiftPoints') && isfield(IKp1,'ExtraLocalShiftAmp')
			        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.ExtraLocalShiftPoints, IKp1.ExtraLocalShiftAmp);
			    end
			    if isfield(IKp1,'ExtraLocalShift2Points') && isfield(IKp1,'ExtraLocalShift2Amp')
			        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.ExtraLocalShift2Points, IKp1.ExtraLocalShift2Amp);
			    end
		    Factor(LDEUseI>=TailIKp.Highist) = tailSlope;

end
    
if ~strcmp(IKp1.Mode,'threesegments')
    if isfield(IKp1,'EarlyShiftPoints') && isfield(IKp1,'EarlyShiftAmp')
        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.EarlyShiftPoints, IKp1.EarlyShiftAmp);
    end
    if isfield(IKp1,'LocalShiftPoints') && isfield(IKp1,'LocalShiftAmp')
        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.LocalShiftPoints, IKp1.LocalShiftAmp);
    end
    if isfield(IKp1,'ExtraLocalShiftPoints') && isfield(IKp1,'ExtraLocalShiftAmp')
        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.ExtraLocalShiftPoints, IKp1.ExtraLocalShiftAmp);
    end
    if isfield(IKp1,'ExtraLocalShift2Points') && isfield(IKp1,'ExtraLocalShift2Amp')
        Factor = Factor + LocalFactorShift(LDEUseI, IKp1.ExtraLocalShift2Points, IKp1.ExtraLocalShift2Amp);
    end
end
LDEUseOut = LDEUseI .* Factor; % multiplicative saturation
end

function y = sigmod(x)
y = exp(x)./(1+exp(x));

end

function TailIKp = frozenTailParams(IKp1)
TailIKp = IKp1;
TailIKp.Thrsld2 = getfieldWithDefault(IKp1, 'TailThrsld2', IKp1.Thrsld2);
TailIKp.Highist = getfieldWithDefault(IKp1, 'TailHighist', IKp1.Highist);
TailIKp.Slope = getfieldWithDefault(IKp1, 'TailSlope', IKp1.Slope);
TailIKp.down1 = getfieldWithDefault(IKp1, 'TailDown1', IKp1.down1);
TailIKp.IntaH = getfieldWithDefault(IKp1, 'TailIntaH', IKp1.IntaH);
TailIKp.IntaL = getfieldWithDefault(IKp1, 'TailIntaL', IKp1.IntaL);
end

function value = getfieldWithDefault(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function [progress, slope] = secondSegmentProgressAndSlope(x, IKp1, down1, down2)
aL = IKp1.IntaL;
aH = IKp1.IntaH;
highaa = sigmod(aH);
lowaa = sigmod(-aL);
xxRef = 1:220;
xxRefId2 = xxRef(xxRef<IKp1.Highist & xxRef>=IKp1.Thrsld2);
xxaaRef = (xxRefId2 - IKp1.Thrsld2)/(IKp1.Highist - IKp1.Thrsld2) * (aH+aL) -aL;
sigmaaRef = sigmod(xxaaRef);
sigmRescleaaRef = (sigmaaRef - lowaa)/(highaa-lowaa);
refMin = min(sigmRescleaaRef,[],'all');
refRange = max(sigmRescleaaRef - refMin,[],'all');

xxaa = (x - IKp1.Thrsld2)/(IKp1.Highist - IKp1.Thrsld2) * (aH+aL) -aL;
sigmaa = sigmod(xxaa);
sigmRescleaa = (sigmaa - lowaa)/(highaa-lowaa);
progress = (sigmRescleaa - refMin) / refRange * down2 + down1;
if nargout > 1
    dzdx = (aH+aL)/(IKp1.Highist - IKp1.Thrsld2);
    slope = sigmaa .* (1-sigmaa) * dzdx / (highaa-lowaa) / refRange * down2;
end
end

function shift = LocalFactorShift(x, points, amp)
shift = zeros(size(x));
leftInd = x>=points(1) & x<=points(2);
rightInd = x>points(2) & x<=points(3);
shift(leftInd) = amp * 0.5 .* (1 - cos(pi*(x(leftInd) - points(1))/(points(2) - points(1))));
shift(rightInd) = amp * 0.5 .* (1 + cos(pi*(x(rightInd) - points(2))/(points(3) - points(2))));
end

function Factor = smoothMultisigmoidJoins(x, Factor, IKp1, down1, down2)
halfWidths = IKp1.SmoothJoinHalfWidth;
if isscalar(halfWidths)
    halfWidths = repmat(halfWidths, 1, 3);
end
halfWidths = halfWidths(:).';
firstFactor = firstMultisigmoidFactor(x, IKp1, down1);
secondFactor = secondMultisigmoidFactor(x, IKp1, down1, down2);
tailFactor = IKp1.Slope * ones(size(x));
Factor = ones(size(x));

firstCore = x >= IKp1.Thrsld1+halfWidths(1) & x <= IKp1.Thrsld2-halfWidths(2);
secondCore = x >= IKp1.Thrsld2+halfWidths(2) & x <= IKp1.Highist-halfWidths(3);
tailCore = x >= IKp1.Highist+halfWidths(3);
Factor(firstCore) = firstFactor(firstCore);
Factor(secondCore) = secondFactor(secondCore);
Factor(tailCore) = tailFactor(tailCore);

Factor = smoothBlend(Factor, firstFactor, x, IKp1.Thrsld1, halfWidths(1));
joinInd = x > IKp1.Thrsld2-halfWidths(2) & x < IKp1.Thrsld2+halfWidths(2);
if any(joinInd,'all')
    w = compactSmoothStep((x(joinInd) - (IKp1.Thrsld2-halfWidths(2)))/(2*halfWidths(2)));
    Factor(joinInd) = (1-w).*firstFactor(joinInd) + w.*secondFactor(joinInd);
end
tailInd = x > IKp1.Highist-halfWidths(3) & x < IKp1.Highist+halfWidths(3);
if any(tailInd,'all')
    w = compactSmoothStep((x(tailInd) - (IKp1.Highist-halfWidths(3)))/(2*halfWidths(3)));
    Factor(tailInd) = (1-w).*secondFactor(tailInd) + w.*tailFactor(tailInd);
end
end

function Factor = firstMultisigmoidFactor(x, IKp1, down1)
bL = IKp1.IntbL;
bH = IKp1.IntbH;
xx = (x - IKp1.Thrsld1)/(IKp1.Thrsld2 - IKp1.Thrsld1) * (bH+bL) - bL;
sigm = sigmod(xx);
high = sigmod(bH);
low = sigmod(-bL);
progress = (sigm - low)/(high-low) * down1;
Factor = 1 + progress * (IKp1.Slope - 1);
end

function Factor = secondMultisigmoidFactor(x, IKp1, down1, down2)
aL = IKp1.IntaL;
aH = IKp1.IntaH;
xx = (x - IKp1.Thrsld2)/(IKp1.Highist - IKp1.Thrsld2) * (aH+aL) - aL;
sigm = sigmod(xx);
high = sigmod(aH);
low = sigmod(-aL);
progress = (sigm - low)/(high-low) * down2 + down1;
Factor = 1 + progress * (IKp1.Slope - 1);
end

function y = smoothBlend(y0, y1, x, center, halfWidth)
y = y0;
if halfWidth <= 0
    return;
end
blendInd = x > center-halfWidth & x < center+halfWidth;
if any(blendInd,'all')
    w = compactSmoothStep((x(blendInd) - (center-halfWidth))/(2*halfWidth));
    y(blendInd) = (1-w).*y0(blendInd) + w.*y1(blendInd);
end
end

function w = compactSmoothStep(t)
w = zeros(size(t));
w(t >= 1) = 1;
midInd = t > 0 & t < 1;
if any(midInd,'all')
    a = exp(-1 ./ t(midInd));
    b = exp(-1 ./ (1 - t(midInd)));
    w(midInd) = a ./ (a + b);
end
end

function Factor = smoothT2Quintic(x, Factor, IKp1, down1, down2)
halfWidth = IKp1.SmoothT2QuinticWidth;
leftT = max(IKp1.Thrsld1, IKp1.Thrsld2 - halfWidth);
rightT = min(IKp1.Highist, IKp1.Thrsld2 + halfWidth);
smoothInd = x >= leftT & x <= rightT;
if rightT <= leftT || ~any(smoothInd,'all')
    return;
end
[y0, dy0, ddy0] = multisigmoidRawFactorDerivatives(leftT, IKp1, down1, down2);
[y1, dy1, ddy1] = multisigmoidRawFactorDerivatives(rightT, IKp1, down1, down2);
dx = rightT - leftT;
t = ((x(smoothInd) - leftT) / dx).';
t = t(:).';
T = [ones(size(t)); t; t.^2; t.^3; t.^4; t.^5];
c0 = y0;
c1 = dy0 * dx;
c2 = 0.5 * ddy0 * dx^2;
r1 = y1 - (c0 + c1 + c2);
r2 = dy1 * dx - (c1 + 2*c2);
r3 = ddy1 * dx^2 - 2*c2;
c5 = 0.5 * (r3 + 12*r1 - 6*r2);
c4 = 7*r2 - 15*r1 - r3;
c3 = 10*r1 - 4*r2 + 0.5*r3;
coeff = [c0; c1; c2; c3; c4; c5];
Factor(smoothInd) = sum(coeff .* T, 1);
end

function [factor, slope, curvature] = multisigmoidRawFactorDerivatives(x, IKp1, down1, down2)
aL = IKp1.IntaL; bL = IKp1.IntbL;
aH = IKp1.IntaH; bH = IKp1.IntbH;
if x <= IKp1.Thrsld2
    high = sigmod(bH);
    low = sigmod(-bL);
    dzdx = (bH+bL)/(IKp1.Thrsld2 - IKp1.Thrsld1);
    z = (x - IKp1.Thrsld1) * dzdx - bL;
    sigm = sigmod(z);
    scale = down1/(high-low);
else
    high = sigmod(aH);
    low = sigmod(-aL);
    dzdx = (aH+aL)/(IKp1.Highist - IKp1.Thrsld2);
    z = (x - IKp1.Thrsld2) * dzdx - aL;
    sigm = sigmod(z);
    scale = down2/(high-low);
end
progress = (sigm - low) * scale;
if x > IKp1.Thrsld2
    progress = progress + down1;
end
slopeProgress = sigm .* (1-sigm) * dzdx * scale;
curvatureProgress = sigm .* (1-sigm) .* (1 - 2*sigm) * dzdx^2 * scale;
factor = 1 + progress * (IKp1.Slope - 1);
slope = slopeProgress * (IKp1.Slope - 1);
curvature = curvatureProgress * (IKp1.Slope - 1);
end
