%% Validate setup and post-process setup

% ------------------------------ SYSTEM -----------------------------------

% Swing equation
% Check sizes for inputs to swing equation
assert( isequal(size(M),[n,n]) )
assert( isequal(size(D),[n,n]) )
assert( isequal(size(L),[n,n]) )
% Check symmetry for inertia and damping constant
assert( isdiag(M) )
assert( isdiag(D) )
% Check PSD for power flow linearization
if ~isPsd(L)
    warning('L is not PSD...')
end

% Training and test data
% Check machines are within size of system
assert( max(trainingMachines)<=n & min(trainingMachines)>0 )
assert( max(testMachines)<=n & min(testMachines)>0 )
% Define the m machines that you have data for, and its corresponding
% selection matrix
I_N = eye(n);
S_M = I_N(trainingMachines,:);
% Define sizes of machines
numTrainingMachines = numel(trainingMachines);
numTestMachines = numel(testMachines);

% -------------------------- SIMULATION -----------------------------------

% Input time
% Check prediction times is subset of training
assert( all(ismember(predictionTimesIdx,trainingTimesIdx)) )
% Define training times
trainingTimes = t0(trainingTimesIdx);
numTrainTimeSamples = numel(trainingTimes);
% Define test times
testTimes = t0(testTimesIdx);
numTestTimeSamples = numel(testTimes);
% Test prediction times
predTimes = trainingTimes(ismember(trainingTimesIdx,predictionTimesIdx));
numPredTimeSamples = numel(predTimes);

% Input p(t)
assert( isdiag(COV_P) ) % Assume is IID
assert( isequal(size(COV_P),[n n]) ) % There's an input for each generator


% ------------------------------ NOISE ------------------------------------

% Define parameters for (trivial) Gaussian noise
sigmaNoise = 0;
if includeNoise
    assert(~isempty(varianceNoise_per_second))
    assert(varianceNoise_per_second~=0)
    assert(isscalar(varianceNoise_per_second))
    includeNoise = true;
    sigmaNoise = sqrt(varianceNoise_per_second)*deltaT;
end

% Define parameters for random large noise
% TO DO: There is definitely more efficient way to implement this
if includeRandomLargeNoise
    randomLargeNoise_InjectionIndices = struct();
    for idx = 1:numel(randomLargeNoise_MachinesAffected)
        % Ensure that all prediction times and training times will get
        % noise with priority
        predTimesIdx_forNoise = predictionTimesIdx;
        trainTimesIdx_forNoise = trainingTimesIdx( ...
            ~ismember(trainingTimesIdx,predictionTimesIdx));
        allTimesIdx_forNoise = find(~ismember(1:numel(t0), ...
            [predTimesIdx_forNoise; trainTimesIdx_forNoise]));
        %
        predTimesIdx_forNoise_SCRAMBLED = ...
            predTimesIdx_forNoise( ...
            randperm(numel(predTimesIdx_forNoise)) ...
            );
        trainTimesIdx_forNoise_SCRAMBLED = ...
            trainTimesIdx_forNoise( ...
            randperm(numel(trainTimesIdx_forNoise)) ...
            );
        allTimesIdx_forNoise_SCRAMBLED = ...
            allTimesIdx_forNoise( ...
            randperm(numel(allTimesIdx_forNoise)) ...
            );
        %
        combinedTimeIndices_forNoise_SCRAMBLED = [ ...
            predTimesIdx_forNoise_SCRAMBLED(:); ...
            trainTimesIdx_forNoise_SCRAMBLED(:); ...
            allTimesIdx_forNoise_SCRAMBLED(:) ...
            ];
        % Select times with error (assuming unsampled signal)
        randomLargeNoise_InjectionIndices. ...
            ("Machine" + randomLargeNoise_MachinesAffected(idx)) = ...
            combinedTimeIndices_forNoise_SCRAMBLED( ...
            1:randomLargeNoise_NumInjections(idx));
    end
end

% Drift
if includeDriftNoise
    assert(driftNoise_pps==1) % Limitation for now
    assert(driftNoise_deltaT < driftNoise_pps)
    assert(driftNoise_deltaT > deltaT)
end

% Define "bad" machines with non-trivial noise
% When useOnGoodMachines=true, during prediction, the training machines
% with non-trivial noise on them will be excluded; BAD_MACHINES stores
% these machines that will be excluded
BAD_MACHINES = [];
if includeFalseInjection
    BAD_MACHINES = [BAD_MACHINES falseInjection_MachinesAffected];
end
if includeRandomLargeNoise
    BAD_MACHINES = [BAD_MACHINES randomLargeNoise_MachinesAffected];
end
if includeDriftNoise
    BAD_MACHINES = [BAD_MACHINES driftNoise_MachinesAffected];
end
BAD_MACHINES = unique(BAD_MACHINES);
assert(all(ismember(BAD_MACHINES,trainingMachines)))

% ------------------------- PRE-PROCESSING --------------------------------

% Set up filter params
if inputFilter
    F1 = freqsFilt(1);
    F2 = freqsFilt(2);
    if F1==0
        FILTER = designfilt('lowpassiir','FilterOrder',8, ...
            'HalfPowerFrequency',F2, ...
            'SampleRate',1/deltaT);
    else
        FILTER = designfilt('bandpassiir','FilterOrder',8, ...
            'HalfPowerFrequency1',F1,'HalfPowerFrequency2',F2, ...
            'SampleRate',1/deltaT);
    end
end

% ---------------------------- INVERSE ------------------------------------

if useWoodburyMatIdentityForInverse && ~includeNoise
    error(['When using woodberry matrix identity for calculating the ' ...
        'inverse of a large matrix, need to set includeNoise=true. The ' ...
        'inverse technique requires the use of a sum, which in our ' ...
        'case, is only invertible when noise is added (because the sum ' ...
        'is with the noise variance matrix.'])
end

% ---------------------------- LEARNING -----------------------------------

switch optimizerLoss
    case {'L2','L1'}
    case 'L1_MASK'
        assert(exist('BETA','var'))
    otherwise
        error('Optimizer loss function not supported.')
end

%% Simulate

% System is:
%    M*w'(t) + D*w(t) + L*theta(t) = p(t)
% => M * theta''(t) + D * theta'(t) + L * theta(t) = p(t)
%
% Using subsitution theta'' = (theta')', we get:
%
%    | M | D | |theta'|'   | 0 | L | |theta'|   | p |
%    |--- ---|*|------|  + |--- ---|*|------| = |---|
%    | 0 | I | |theta |    |-I | 0 | |theta |   | 0 |
%
%    |  w  |   |theta'|   | I | 0 | |theta'|   |       | | p |
%    |-----| = |------| = |--- ---|*|------| + |   0   |*|---|
%    |theta|   |theta |   | 0 | I | |theta |   |       | | 0 |
%
%    w = theta'

% Generate data using state-space (SS) simulation
% x' = A*x + B*u
% y  = C*x + D*u
A_ss = inv( [M,D;zeros(n),eye(n)] )*[zeros(n),-L;eye(n),zeros(n)]; %#ok<MINV>
B_ss = inv( [M,D;zeros(n),eye(n)] );
C_ss = eye(2*n);
D_ss = zeros(2*n);
sys_ss = ss(A_ss,B_ss,C_ss,D_ss);
% Create input
p = ( sqrt(COV_P) * randn(n,numel(t0)) ).';
p_vec = [p zeros(size(p))];
% Create discrete system
sys_d = c2d(sys_ss,deltaT);
Ad = sys_d.A;
Bd = sys_d.B;
Cd = sys_d.C;
Dd = sys_d.D;
% Simulate
x = zeros(2*n,numel(t0));
x(:,1) = zeros(2*n,1);
for cnt = 1:numel(t0)-1
    x(:,cnt+1)  = Ad*x(:,cnt) + Bd*(p_vec(cnt,:).');
end
% Save
x = x.';
t = t0;
% Process
y = C_ss*(x.') + D_ss*(p_vec.');
w = y(1:n,:).'; % w is speed
theta = y((n+1):(2*n),:).'; % theta is angle
% Save time
numt = numel(t);

%% Perform eigendeomposition for change of variables

% The original system: 
%   M*w'(t) + D*w(t) + L*theta(t) = p(t)
% Becomes:
%   y''(t) + gamma*y'(t) + LAMBDA*y(t) = x(t)
% 
% Or equivalently, the original transfer function (with w as the state):
%   H(s) = s*(s^2*M + s*D + L)^-1
% Becomes (with y' as the state):
%   H(s) = s*M^(-1/2)*V*(s^2*I + s*gamma*I + LAMBDA)^-1*V^T*M^(-1/2)
%
% Such that V and D are eigendecompositions of Lm.
Lm = (M^(-1/2))*L*(M^(-1/2));
if ~isPsd(Lm)
    warning('Lm is not PSD.')
end
[V,LAMBDA] = eig(Lm);
lambda = diag(LAMBDA);

% Find gamma which is estimate of ratio beteen inertia M and damping
% constant
gamma = mean(diag(D))/mean(diag(M));
gamma_all = diag(D./M);
if max(abs(gamma-gamma_all)./gamma) > 0.01
    warning('Assumption of uniform damping is violated.')
end

%% Perform reduction of eigenspace
[lambda,V] = reduceEigenspace(lambda,V,gamma,freqs);
n_reduced = numel(lambda);

%% Define covariance kernel for training

% In general, the kernel function looks like:
%   E[ var1(tvar1)*var2(tvar2) ] =
%   E[ var1( t+T )*var2(  t  ) ] =
%   M^-1/2*V*(A dot K(T))*V^T*M^-1/2
%
% Here, we want to get the kernel for the data to be trained on (i.e. the
% measured data), where T = 0;
T = 0;
KT = getKernel(lambda,T,gamma);


%% Get training data

% Variable of interest is speed
z_unsamp = w;
trainingZ_unsamp = z_unsamp(:,trainingMachines);
trainingZ_unsamp_noNoise = z_unsamp(:,trainingMachines);

% Generate falsely injected data using state-space (SS) simulation
if includeFalseInjection
    % Copy actual mpc to falseInjection it
    mpc_falseInjection = mpc;
    % -------------------- Falsify network parameters ---------------------
    % Perburb MPC loads and branches
    n_buses = size(mpc.bus,1);
    num_BAD_LOADS = round(n_buses*falseInjection_GB_percentLoads);
    BAD_LOADS = randperm(n_buses,num_BAD_LOADS);
    col_idxG = 5;
    col_idxB = 6;
    meanG = mean(mpc_falseInjection.bus(:,col_idxG)); % For G is 0
    meanB = mean(mpc_falseInjection.bus(:,col_idxB)); % For B is zero
    mpc_falseInjection.bus(BAD_LOADS,col_idxG) = ...
        (mpc_falseInjection.bus(BAD_LOADS,col_idxG) + meanG) .* ...
        abs(falseInjection_G_perturbation_mult + randn(num_BAD_LOADS,1)* ...
        falseInjection_G_perturbation_var);
    mpc_falseInjection.bus(BAD_LOADS,col_idxB) = ...
        (mpc_falseInjection.bus(BAD_LOADS,col_idxB) + meanB) .* ...
        abs(falseInjection_B_perturbation_mult + randn(num_BAD_LOADS,1)* ...
        falseInjection_B_perturbation_var);
    % Re-compute system linearization for L, as well as generator values
    [L_falseInjection,M_falseInjection,~, ...
        MPC_falseInjection,P_load_falseInjection,Q_load_falseInjection] = ...
        calc_L_M_D(mpc_falseInjection,n,fileName);
    % ------------------ Falsify generator parameters ---------------------
    % Define M, D, and L by perturbing it
    M_falseInjection = M_falseInjection.*(1 + unifrnd(-1,1,n,1)* ...
        falseInjection_systemPerturbation);
    D_falseInjection = diag((0.0057/mean(diag(M_falseInjection))) * ...
        diag(M_falseInjection));
    L_falseInjection = L_falseInjection.*(1 + unifrnd(-1,1,n)* ...
        falseInjection_systemPerturbation);
    % ----------------------- Falsify input p(t) --------------------------
    % Create input data
    COV_P_falseInjection = 0.01*M_falseInjection^2;
    switch falseInjection_p
        case 'same'
            p_falseInjection = p;
            p_vec_falseInjection = p_vec;
        case 'diff_Gaussian'
            p_falseInjection = (sqrt(COV_P_falseInjection)*randn(n,numt)).';
            p_vec_falseInjection = [p_falseInjection zeros(size(p_falseInjection))];
    end
    % -------------------- Simulate for false data ------------------------
    % Create SS
    A_ss_falseInjection = ...
        inv( [M_falseInjection,D_falseInjection;zeros(n),eye(n)] )* ...
        [zeros(n),-L_falseInjection;eye(n),zeros(n)]; %#ok<MINV>
    B_ss_falseInjection = ...
        inv( [M_falseInjection,D_falseInjection;zeros(n),eye(n)] );
    C_ss_falseInjection = eye(2*n);
    D_ss_falseInjection = zeros(2*n);
    %
    sys_ss_falseInjection = ss(A_ss_falseInjection,B_ss_falseInjection,...
        C_ss_falseInjection,D_ss_falseInjection);
    % Create discrete system
    sys_d_falseInjection = c2d(sys_ss_falseInjection,deltaT);
    Ad_falseInjection = sys_d_falseInjection.A;
    Bd_falseInjection = sys_d_falseInjection.B;
    Cd_falseInjection = sys_d_falseInjection.C;
    Dd_falseInjection = sys_d_falseInjection.D;
    % Simulate
    x_falseInjection = zeros(2*n,numt);
    x_falseInjection(:,1) = zeros(2*n,1);
    input_falseInjection = p_vec_falseInjection.';
    for cnt_falseInjection = 1:numt-1
        x_falseInjection(:,cnt_falseInjection+1)  = ...
            Ad_falseInjection*x_falseInjection(:,cnt_falseInjection) + ...
            Bd_falseInjection*input_falseInjection(:,cnt_falseInjection);
    end
    % Save
    x_falseInjection = x_falseInjection.';
    % Process
    y_falseInjection = C_ss_falseInjection*(x_falseInjection.') + ...
        D_ss_falseInjection*(p_vec_falseInjection.');
    w_falseInjection = y_falseInjection(1:n,:).';
    theta_falseInjection = y_falseInjection((n+1):(2*n),:).';
    % Variable of interest is speed
    z_unsamp_falseInjection = w_falseInjection;
end

% Measurement noise
% (e.g. fake injections,random meter mis-readings,additive noise)
if includeNoise || ...
        includeFalseInjection || ...
        includeRandomLargeNoise
    % Before noise
    figure,hold on,plot(t0,trainingZ_unsamp(1:numel(t0),:),'g')
    lgd = cellstr("Noiseless w" + trainingMachines);
    % Add false injection
    idxfalseInjectionList = trainingMachines( ...
        ismember(trainingMachines,falseInjection_MachinesAffected));
    idxfalseInjection = 1;
    if includeFalseInjection
        for idx = 1:numel(trainingMachines)
            machine = trainingMachines(idx);
            if any(machine==falseInjection_MachinesAffected)
                trainingZ_unsamp(:,idx) = z_unsamp_falseInjection(:, ...
                    idxfalseInjectionList(idxfalseInjection));
                idxfalseInjection = idxfalseInjection + 1;
                % Plot false injection
                plot(t0,trainingZ_unsamp(1:numel(t0),idx),'-r')
                lgd = [lgd cellstr("False Injection w" + machine)]; %#ok<AGROW>
            end
        end
    end
    % Add large random noise
    if includeRandomLargeNoise
        for idx = 1:numel(trainingMachines)
            machine = trainingMachines(idx);
            if any(machine==randomLargeNoise_MachinesAffected)
                timeIndices_unsamp_forNoise =  ...
                    randomLargeNoise_InjectionIndices. ...
                    ("Machine" + machine);
                trainingZ_unsamp(timeIndices_unsamp_forNoise,idx) = ...
                    sign(randn(numel(timeIndices_unsamp_forNoise),1)) ...
                    .* (randomLargeNoise_Magnitude + ...
                    randn(numel(timeIndices_unsamp_forNoise),1)* ...
                    randomLargeNoise_Var) + ...
                    trainingZ_unsamp(timeIndices_unsamp_forNoise,idx);
                % Plot large random noise
                plot(t0,trainingZ_unsamp(1:numel(t0),idx),'-r')
                lgd = [lgd cellstr("Large err w" + machine)]; %#ok<AGROW>
            end
        end
    end
    % Add trivial Gaussian additive noise
    if includeNoise
        trainingZ_unsamp = trainingZ_unsamp + randn(size(trainingZ_unsamp)) ...
            * sigmaNoise;
    end
    % Finalize plotting
    legend(lgd{:},'ItemHitFcn',@cb_legend)
    title('Unfiltered and Unsampled signal')
end


% Filter
if inputFilter
    z_unsamp = filtfilt(FILTER,z_unsamp);
    trainingZ_unsamp = filtfilt(FILTER,trainingZ_unsamp);
    trainingZ_unsamp_noNoise = filtfilt(FILTER,trainingZ_unsamp_noNoise);
end

% Downsample
z = z_unsamp(trainingTimesIdx,:);
trainingZ = trainingZ_unsamp(trainingTimesIdx,:);
trainingZ_noNoise = trainingZ_unsamp_noNoise(trainingTimesIdx,:);

% Sampling noise
% (e.g. PMU clock drift)
if includeDriftNoise
    % Before noise
    idxTrainDrift = ismember(trainingMachines,driftNoise_MachinesAffected);
    figure,plot(trainingTimes,trainingZ(:,idxTrainDrift),'-b')
    hold on
    % Calculate drift time for modeling periodic delay
    t0_drift = t0-mod(t0*driftNoise_deltaT,driftNoise_pps*driftNoise_deltaT);
    % Calculate drift time indices
    trainingTimesIdx_drift = zeros(size(trainingTimesIdx));
    for idxDrift = 1:numel(trainingTimesIdx_drift)
        [~,trainingTimesIdx_drift(idxDrift)] = ...
            min(abs( t0_drift(trainingTimesIdx(idxDrift)) - t0 ));
    end
    % Implement drift in sampled data
    for idx = 1:numel(trainingMachines)
        machine = trainingMachines(idx);
        if any(machine==driftNoise_MachinesAffected)
            trainingZ(:,idx) = ...
                trainingZ_unsamp(trainingTimesIdx_drift,idx);
        end
    end
    % Plot drift
    plot(trainingTimes,trainingZ(:,idxTrainDrift),'-r')
    legend(["Noisy "+trainingMachines(idxTrainDrift) ...
        "DRIFTED Noisy "+trainingMachines(idxTrainDrift)], ...
        'ItemHitFcn',@cb_legend)
    title('PMU Drift (Filtered and Sampled)')
end

% Visualize downsampled and filtered signal
figure
plot(trainingTimes,trainingZ(:,~ismember(trainingMachines,BAD_MACHINES)),'g')
hold on
if ~isempty(BAD_MACHINES)
    plot(trainingTimes,trainingZ(:,ismember(trainingMachines,BAD_MACHINES)),'r')
    plot(trainingTimes,trainingZ_noNoise(:,ismember(trainingMachines,BAD_MACHINES)),'b')
end
idxMachBad = ismember(trainingMachines,BAD_MACHINES);
legend(["Machine"+trainingMachines(~idxMachBad) ...
    "Machine"+trainingMachines(idxMachBad) ...
    "Machine"+trainingMachines(idxMachBad)+" (if un-altered)"], ...
    'ItemHitFcn',@cb_legend)
title('Filtered and Sampled signal')

% Define covariance of sample
% NOTE: This is different from MATLAB's shipped "covariance" function
% because it assumes mean is 0 instead of subtracting it
trainingC0_sample = (trainingZ.')*trainingZ/size(trainingZ,1);

%% Get test data

% Variable of interest is speed
testZ_unsamp = w(:,testMachines);

% Filter
if inputFilter
    testZ_unsamp = filtfilt(FILTER,testZ_unsamp);
end

% Downsample
testZ = testZ_unsamp(testTimesIdx,:);
