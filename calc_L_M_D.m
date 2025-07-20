function [L,M,D, MPC, P_load,Q_load,busesInclude,busesExclude, varargout]  ...
    = calc_L_M_D(mpc,n,fileName)
% Calculates the linearized system from MATPOWER mpc structure. If missing
% system information, randomly generate parameter values.
%
% Inputs:
% 'mpc'          - MATPOWER model structure
% 'n'            - Number of generators
% 'fileName'     - Name of file that MPC represents, used to decide how to deal
%                  with uniform damping
%
% Outputs:
% 'L'            - Negative Jacobian matrix of power flow equations after Kron
%                  reduction
% 'M'            - Diagonal matrix of generator inertia
% 'D'            - Diagonal matrix of generator damping coefficients
% 'P_load'       - Real power at loads
% 'Q_load'       - Reactive power at loads
% 'busesInclude' - Buses with generators
% 'busesExclude' - Buses without generators
%
% NOTE:
% - This function was written with references to the open source code from
%   the paper "Inferring Power System Frequency Oscillations using Gaussian
%   Processes" by Jalali et. al. [1]
% - The referenced Github is: https://github.com/manajalali/GP4GridDynamics

varargout = {};

% Reorder
MPC = ext2int(mpc);
assert(isequal(MPC.order.gen.e2i,MPC.order.gen.i2e)) % Gen order is
                                                     % preserved, no need
                                                     % to modify Ls and M
% Load system properties
P_load = MPC.bus(:,3)/MPC.baseMVA;
Q_load = MPC.bus(:,4)/MPC.baseMVA;
busesInclude = MPC.gen(:,1);
busesExclude = setdiff(1:length(P_load),busesInclude);
if isfield(MPC,'Ls')
    % Solve for linearized L from Y
    Y = full(makeYbus(MPC));
    yg = -1i*1./(MPC.Ls(1:n,2));
    Y = Y - diag(P_load-Q_load*1i);
    %
    Y_GL = Y(busesInclude,busesExclude);
    Y_LL = Y(busesExclude,busesExclude);
    Y_G_tilda = Y(busesInclude,busesInclude)+diag(yg);
    %
    YA = Y_G_tilda - Y_GL*(eye(size(Y_LL,1))/Y_LL)*Y_GL.';
    YA = (eye(length(YA))/YA)*diag(yg);
    Y = conj(-diag(yg)*YA);
    %
    BB=real(Y);
    L2_var=zeros(size(BB));
    for i=1:size(BB,1)
        for j=1:size(BB,2)
            if i==j
                L2_var(i,j)=-sum(BB(i,:))+BB(i,j);
            else
                L2_var(i,j)=BB(i,j);
            end
        end
    end
    L = L2_var;
    varargout = [varargout {Y,yg,Y_GL,Y_LL,Y_G_tilda,YA,BB,L2_var}];
else
    % Solve for linearized L from Jacobian
    J0 = full(makeJac(MPC,'fullJac'));
    J = J0(1:length(P_load),1:length(P_load));
    %
    L = J(busesInclude,busesInclude) ...
        - ...
        J(busesInclude,busesExclude)* ...
        inv(J(busesExclude,busesExclude))*...
        J(busesExclude,busesInclude); %#ok<MINV>

    % Make symmetric
    L = (L+L.')/2;
    varargout = [varargout {J0,J}];
end


% Calculate M (inertia)
if isfield(MPC,'M')
    % If contains information about inertia, use it
    M = diag(MPC.M(:,2));
else
    % Randomly generate data representing M
    % Assume H is typically 1-10 seconds
    H = rand(n,1)*9 + 1;
    f_m = 60; % [Hz]
    M = diag( H/(pi*f_m) );

    % Output information used to calculate M
    varargout = [varargout {H,f_m}];
end

% Calculate D (damping) with homogenous assumption of uniform damping
if strcmpi(fileName,'case300.m')
    % Repicate calculations from [1]
    m = diag(M);
    d = (0.0057/mean(m)) * m;
    D = diag(d);

    % Output information used to calculate M
    varargout = [varargout {m,d}];
else
    % Based on the model having ratios lying "within the relatively narrow
    % range of [0.19,0.4]" in [1], we take the average here
    D = M*mean([0.19,0.4]);
end
end