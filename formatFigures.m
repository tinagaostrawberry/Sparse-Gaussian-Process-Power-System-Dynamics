function formatFigures(f,rmseVal,mach,nonRobust)
% Format figures for display
%
% Inputs:
% 'f'       - Figure handle
% 'rmseVal' - Min-max normalized root mean square error
% 'mach'    - Machine number

% Modify plot size
f.Position = [360.0000  408.3333  910.3333  189.3333];

% Axes
xlabel('Time [s]')
ylabel('Speed [rad/s]')

% Font
fontsize(15,"points")

% Modify markers
a = f.CurrentAxes;
c = a.Children;
c(2).MarkerSize = 15;
c(3).LineWidth = 2.5;

% Title
if (nargin==3) || ~nonRobust
    if isempty(rmseVal)
        title("Aggregate Group of Machine "+mach)
    elseif isempty(mach)
        title("Proposed Robust GP Prediction (RMSE: " + rmseVal + ")")
    else
        title("Proposed Robust GP Prediction of Machine " + mach + ...
            " (RMSE: " + rmseVal + ")")
    end
else
    title("Non-robust GP Prediction (RMSE: " + rmseVal + ")")
end

end