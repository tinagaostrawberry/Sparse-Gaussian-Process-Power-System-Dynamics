function [rmseVal,mach] = getFigureTitleInfo(f,lossFuncChar)
% Get RMSE and machien number from title of figure
%
% Input:
% 'f'    - Figure handle

% Outputs:
% 'rmse' - Mean-max normalized root mean square error
% 'mach' - Machine number being predicted

tit = get(get(f.CurrentAxes,'title'),'string');
mach = str2double(strrep(strrep(tit{1},'Machine ',''),lossFuncChar,''));
rmseVal = str2double(strrep(tit{2},'Min-max-normalized RMSE: ',''));

end