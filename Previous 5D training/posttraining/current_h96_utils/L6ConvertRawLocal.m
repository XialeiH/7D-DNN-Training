%% L6 convertion function: L6EUse has the same dimensionality as C;
% L6pars is a cell containing parameters for the linear function

function y = L6ConvertRawLocal(x, L6pars)
if iscell(L6pars{end}) && numel(L6pars{end}) >= 4 && strcmp(L6pars{end}{1}, 'postlocalbump')
    basePars = L6pars{end}{2};
    bumpPoints = L6pars{end}{3};
    bumpAmp = L6pars{end}{4};
    y = L6Convert(x, basePars) .* LocalBump(x, bumpPoints, bumpAmp);
    return
end
if iscell(L6pars{end}) && numel(L6pars{end}) >= 4 && strcmp(L6pars{end}{1}, 'weightedblend')
    ratio = L6pars{end}{2};
    y0 = L6Convert(x, L6pars{end}{3});
    y1 = L6Convert(x, L6pars{end}{4});
    y = (1-ratio) .* y0 + ratio .* y1;
    return
end
y = L6pars{1} * (x - L6pars{2});
% Apply the initial linear mapping
if strcmp(L6pars{end},'quadratic')
    % in this case, we will use the following three points to determine
    % an inverse function of quadratic function
    x_data = L6pars{4}; y_data = L6pars{3};
    % Create the matrix A for the system of equations
    A = [x_data.^2, x_data, ones(3, 1)];

    % Solve for the coefficients [a; b; c]
    Quad_coeff = A \ y_data;

    % Display the coefficients
    a = Quad_coeff(1);
    b = Quad_coeff(2);
    c = Quad_coeff(3);
    % fprintf('The quadratic function is y = %.4fx^2 + %.4fx + %.4f\n', a, b, c);
    xAdj = x(x > y_data(1));
    y(y > x_data(1)) = (-b + sqrt(b^2 - 4*a*c + 4*a*xAdj)) / (2*a);

elseif iscell(L6pars{end}) % using cell in the last entry to modify the low part of conversion curve
    % Number of pairs beyond the threshold
    numPairs = (length(L6pars) -3 - 1) / 2;
    if length(L6pars)>=3
        % Arrays for piecewise linear interpolation
        xPoints = [L6pars{3}];  % Starting with the threshold
        yPoints = [L6pars{3}];  % Assuming y = x at the threshold

        % Construct the arrays for interpolation based on the provided rules
        for n = 2:numPairs+1
            yPoints(end+1) = L6pars{2*n+1};  % y values to be mapped
            xPoints(end+1) = L6pars{2*n};    % Corresponding y values after mapping
        end

        % Ensure xPoints are in increasing order along with their corresponding yPoints
        [xPoints, sortIdx] = sort(xPoints, 'ascend');
        yPoints = yPoints(sortIdx);
        IdInter = y > L6pars{3} & y <= xPoints(end);
        IdExter = y > xPoints(end);
        y(IdInter) = interp1(xPoints, yPoints, y(IdInter), 'linear');
        y(IdExter) = interp1(xPoints, yPoints, y(IdExter), 'linear', 'extrap');
    end
    Lowpars = L6pars{end};
    X = Lowpars{1}; % three x points: 0, middle, threshold
    Y = [Lowpars{2}, L6pars{1}*(X(end)-L6pars{2})]; %two points:
    % Create the matrix for the quadratic system
    A = [X(1)^2, X(1), 1;
        X(2)^2, X(2), 1;
        X(3)^2, X(3), 1];

    % Solve for the coefficients [a; b; c]
    coeffs = A \ Y';

    % Define the quadratic function
    quadratic_func = @(x) coeffs(1)*x.^2 + coeffs(2)*x + coeffs(3);
    IdLow = y <= Y(end);
    y(IdLow) = quadratic_func(x(IdLow));
    % quadratic for high end as well
    if length(Lowpars) >= 4
        Xhigh = Lowpars{3}; % three x points: threshold (on the line), two points above
        if length(Lowpars) >= 6
            Xbase = Lowpars{5};
            Ybase = Lowpars{6};
            C = [Xbase(1)^2, Xbase(1), 1;
                Xbase(2)^2, Xbase(2), 1;
                Xbase(3)^2, Xbase(3), 1];
            coeffsbase = C \ Ybase';
            baseline_func_high = @(x) coeffsbase(1)*x.^2 + coeffsbase(2)*x + coeffsbase(3);
            baseline_deriv_high = @(x) 2*coeffsbase(1)*x + coeffsbase(2);

            x0 = Xhigh(1);
            x1 = Xhigh(2);
            y0 = Lowpars{4}(1);
            y1 = baseline_func_high(x1);
            m0 = baseline_deriv_high(x0);
            m1 = baseline_deriv_high(x1);

            IdBridge = x >= x0 & x < x1;
            t = (x(IdBridge) - x0) ./ (x1 - x0);
            h00 = 2*t.^3 - 3*t.^2 + 1;
            h10 = t.^3 - 2*t.^2 + t;
            h01 = -2*t.^3 + 3*t.^2;
            h11 = t.^3 - t.^2;
            y(IdBridge) = h00*y0 + h10*(x1-x0)*m0 + h01*y1 + h11*(x1-x0)*m1;

            IdBaseHigh = x >= x1;
            y(IdBaseHigh) = baseline_func_high(x(IdBaseHigh));
        elseif numel(Lowpars{4}) == 3
            Yhigh = Lowpars{4};
            B = [Xhigh(1)^2, Xhigh(1), 1;
                Xhigh(2)^2, Xhigh(2), 1;
                Xhigh(3)^2, Xhigh(3), 1];

            coeffshigh = B \ Yhigh';

            % Define the quadratic function
            quadratic_func_high = @(x) coeffshigh(1)*x.^2 + coeffshigh(2)*x + coeffshigh(3);
            Idhigh = y >= Yhigh(1);
            y(Idhigh) = quadratic_func_high(x(Idhigh));
        else
            Yhigh = [L6pars{1}*(Xhigh(1)-L6pars{2}), Lowpars{4}]; %two points:
            B = [Xhigh(1)^2, Xhigh(1), 1;
                Xhigh(2)^2, Xhigh(2), 1;
                Xhigh(3)^2, Xhigh(3), 1];

            coeffshigh = B \ Yhigh';

            % Define the quadratic function
            quadratic_func_high = @(x) coeffshigh(1)*x.^2 + coeffshigh(2)*x + coeffshigh(3);
            Idhigh = y >= Yhigh(1);
            y(Idhigh) = quadratic_func_high(x(Idhigh));
        end
    end

%% one more twick to get better Contrast response
    if iscell(Lowpars{end}) 
        ConTreakPar = Lowpars{end};
        if numel(ConTreakPar) >= 3 && strcmp(ConTreakPar{3}, 'localbump')
            TweakFac = LocalBump(x, ConTreakPar{1}, ConTreakPar{2});
        else
            TweakPoints = ConTreakPar{1};
            TweakScale = ConTreakPar{2};
            TweakFac = ContraTreak(x,TweakPoints,TweakScale);
        end
        y = y.*TweakFac;
    end

else % piecewise linear
    % Number of pairs beyond the threshold
    numPairs = (length(L6pars) - 3) / 2;
    if length(L6pars)>=3
        % Arrays for piecewise linear interpolation
        xPoints = [L6pars{3}];  % Starting with the threshold
        yPoints = [L6pars{3}];  % Assuming y = x at the threshold

        % Construct the arrays for interpolation based on the provided rules
        for n = 2:numPairs+1
            yPoints(end+1) = L6pars{2*n+1};  % y values to be mapped
            xPoints(end+1) = L6pars{2*n};    % Corresponding y values after mapping
        end

        % Ensure xPoints are in increasing order along with their corresponding yPoints
        [xPoints, sortIdx] = sort(xPoints, 'ascend');
        yPoints = yPoints(sortIdx);
        IdInter = y > L6pars{3} & y <= xPoints(end);
        IdExter = y > xPoints(end);
        y(IdInter) = interp1(xPoints, yPoints, y(IdInter), 'linear');
        y(IdExter) = interp1(xPoints, yPoints, y(IdExter), 'linear', 'extrap');
    end
end
end

function TweakFac = ContraTreak(x,TweakPoints,TweakScale)
TweakFac = ones(size(x));
upInd = x<=TweakPoints(2) & x>=TweakPoints(1);
lowInd = x<=TweakPoints(3) & x>TweakPoints(2);
TweakFac(upInd) = 1 + TweakScale(1) *sin(pi*(x(upInd) - TweakPoints(1))/(TweakPoints(2) - TweakPoints(1)));
TweakFac(lowInd) = 1 - TweakScale(2) *sin(pi*(x(lowInd) - TweakPoints(2))/(TweakPoints(3) - TweakPoints(2)));

end

function TweakFac = LocalBump(x,TweakPoints,TweakScale)
TweakFac = ones(size(x));
leftInd = x>=TweakPoints(1) & x<=TweakPoints(2);
rightInd = x>TweakPoints(2) & x<=TweakPoints(3);
TweakFac(leftInd) = 1 + TweakScale * 0.5 .* ...
    (1 - cos(pi*(x(leftInd) - TweakPoints(1))/(TweakPoints(2) - TweakPoints(1))));
TweakFac(rightInd) = 1 + TweakScale * 0.5 .* ...
    (1 + cos(pi*(x(rightInd) - TweakPoints(2))/(TweakPoints(3) - TweakPoints(2))));

end

% function L6EUse = L6Convert(C,L6pars)
%
% if length(L6pars)==2
%     L6EUse =  L6pars{1}*(C-L6pars{2});
% elseif length(L6pars) >= 3
%     BentStart = L6pars{3} % starting point to bend
%
%
%     L6EUsePre = L6pars{1}*(C-L6pars{2});
%     L6EUse = L6EUsePre;
%     L6EUse(L6EUse > L6pars{3}) = ...
%         L6pars{3} + (L6EUsePre(L6EUsePre > L6pars{3}) - L6pars{3}) * ...
%         (L6pars{5} - L6pars{3})/(L6pars{4}- L6pars{3});
% else
%     error('Illigal L4->L6 parameters')
% end
%
% end
