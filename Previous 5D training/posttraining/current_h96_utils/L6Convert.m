function y = L6Convert(x, L6pars)
if iscell(L6pars{end}) && numel(L6pars{end}) >= 3 && strcmp(L6pars{end}{1}, 'c1table')
    grid = L6pars{end}{2};
    yGrid = L6pars{end}{3};
    [grid, sortIdx] = sort(grid(:));
    yGrid = yGrid(sortIdx);
    [grid, uniqueIdx] = unique(grid, 'stable');
    yGrid = yGrid(uniqueIdx);
    y = interp1(grid, yGrid, x, 'pchip', 'extrap');
    return
end
if iscell(L6pars{end}) && numel(L6pars{end}) >= 3 && strcmp(L6pars{end}{1}, 'c1smooth')
    basePars = L6pars{end}{2};
    grid = L6pars{end}{3};
    grid = unique(grid(:));
    yGrid = L6ConvertRawLocal(grid, basePars);
    y = interp1(grid, yGrid, x, 'pchip', 'extrap');
    return
end

y = L6ConvertRawLocal(x, L6pars);
end
