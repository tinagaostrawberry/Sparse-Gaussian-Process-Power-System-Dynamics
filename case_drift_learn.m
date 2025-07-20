%% Covariance/correlation matrices - calculate

% Compute time indices at which the signal is temporally shifted for each
% laggy covariance
% Find time index at 1 second
[~,idx_1sec] = min(abs(1-trainingTimes));
% Find the time indices within 1 second that correspond to the set of lags
idx_lagTick = round(idx_1sec/numLagsPerSec);
tIdxShifts = [0 (idx_lagTick:idx_lagTick:((numLagsPerSec-1)*idx_lagTick))];
% Sanity check that all lags are within 1 second
assert(all(trainingTimes(tIdxShifts+1)<1))

% Calculate covariances and kernels for each lag
% (Re-)Initialize training covariance and kernel variable that can hold lags
numCov = numel(tIdxShifts);
trainingC_sample = zeros(numTrainingMachines,numTrainingMachines,numCov);
K = zeros(n_reduced,n_reduced,numCov);
% Compute timestep for training data
train_deltaT = uniquetol(diff(trainingTimes),1e-9);
assert(isscalar(train_deltaT)) % Verify timestep and sampling is uniform
% Iterate through lags
for idxLag = 1:numCov
    tIdxShift = tIdxShifts(idxLag);
    % Calculate covariance w.r.t. each lag
    trainingC_sample(:,:,idxLag) = (trainingZ((1+tIdxShift):end,:).')* ...
        trainingZ(1:(end-tIdxShift),:)/ ...
        size(trainingZ((1+tIdxShift):end,:),1);
    % Calculate kernel w.r.t. each lag
    tLag = train_deltaT*tIdxShift;
    K(:,:,idxLag) = getKernel(lambda,tLag,gamma);
end

%% Covariance/correlation matrices - set up for learning

% Set up for using covariance or correlation
STD_ztraining = zeros(numTrainingMachines,numTrainingMachines,numCov);
STD_z = zeros(n,n,numCov);
STD_z0 = zeros(n,n,numCov);
if optimizerNormalize
    for idxLag = 1:numCov
        tIdxShift = tIdxShifts(idxLag);

        % Find standard deviation
        % ...for NOISY training data
        std_ztraining_lag = abs(diag(trainingC_sample(:,:,idxLag))).^(1/2);
        % ...for test data
        std_ztest_lag = abs(diag( ...
            (z((1+tIdxShift):end,testMachines).')* ...
            z(1:(end-tIdxShift),testMachines)/ ...
            size(z((1+tIdxShift):end,testMachines),1) ...
            )).^(1/2);
        % ... for test and CLEAN training data
        std_z0_lag = abs(diag( ...
            (z((1+tIdxShift):end,:).')* ...
            z(1:(end-tIdxShift),:)/ ...
            size(z((1+tIdxShift):end,:),1) ...
            )).^(1/2);

        % Get standard deviation multiplier
        % ...for noisy training data
        STD_ztraining(:,:,idxLag) = diag(1./std_ztraining_lag);
        % ...for test and noisy training data combined
        std_z_lag = zeros(1,n);
        std_z_lag(trainingMachines) = std_ztraining_lag;
        std_z_lag(testMachines) = std_ztest_lag;
        STD_z(:,:,idxLag) = diag(1./std_z_lag);
        % ...for test and CLEAN training data
        STD_z0(:,:,idxLag) = diag(1./std_z0_lag);
    end
else
    % Set all standard deviation matrices to identity
    STD_ztraining = repmat(eye(numTrainingMachines),1,1,numCov);
    STD_z = repmat(eye(n),1,1,numCov);
    STD_z0 = repmat(eye(n),1,1,numCov);
end

% Get vectorized parameters for learning
c_multiplier = zeros(numCov*numTrainingMachines^2,n_reduced^2);
trainingC_sample_norm = zeros(numTrainingMachines,numTrainingMachines,numCov);
for idxLag = 1:numCov
    % Get parameterized covariance/correlation vector
    KT_lag = K(:,:,idxLag);
    kt_lag = KT_lag(:);
    U_lag = kron(STD_ztraining(:,:,idxLag)*S_M*(M^(-1/2))*V, ...
        STD_ztraining(:,:,idxLag)*S_M*(M^(-1/2))*V);
    c_multiplier( ...
        (idxLag-1)*(numTrainingMachines^2)+(1:numTrainingMachines^2), ...
        1:n_reduced^2) = ...
        U_lag*diag(kt_lag);
    % Get measured covariance/correlation vector
    trainingC_sample_norm(:,:,idxLag) =  STD_ztraining(:,:,idxLag)* ...
        trainingC_sample(:,:,idxLag)*STD_ztraining(:,:,idxLag);
end
% Finalize measured covariance/correlation vector
if includeNoise
    % Subtract trivial Gaussian additive noise
    trainingC_sample_norm = trainingC_sample_norm + ...
        eye(numTrainingMachines)*sigmaNoise^2;
end
c_sample = trainingC_sample_norm(:);

%% MOM optimization

% Equality constraints for keeping A symmetric
if optimizerASymm
    % TO DO: Modify the brute force method below to be elegant solution
    numPairs = (n_reduced^2-n_reduced)/2;
    Aeq_symm = zeros(numPairs,n_reduced^2);
    beq_symm = zeros(numPairs,1);
    A_mat = sym('a',n_reduced);
    A_vec = A_mat(:);
    idxRow = 1;
    for i1 = 1:n_reduced
        for i2 = 1:n_reduced
            if i1 > i2
                idxColPos = find(A_vec==['a' num2str(i1) '_' num2str(i2)]);
                idxColNeg = find(A_vec==['a' num2str(i2) '_' num2str(i1)]);
                Aeq_symm(idxRow,idxColPos) = 1;
                Aeq_symm(idxRow,idxColNeg) = -1;
                idxRow = idxRow + 1;
            end
        end
    end
    assert(idxRow == numPairs+1)
else
    Aeq_symm = [];
    beq_symm = [];
end

% Solve optimization problem with specified loss function
% Rename variables as to names with lag T=0 for optimization
c0_multiplier = c_multiplier;
c0_sample = c_sample;
switch optimizerLoss
    case 'L2'
        % Learn covariance of eigenstates A
        L2LearnA

    case 'L1'
        % Learn covariance of eigenstates A
        L1LearnA   

    case 'L1_MASK'
        % Learn covariance of eigenstates A
        L1LearnA

        % Learn weights for masking out bad data
        % Find summation across lags
        c_multiplier_avg = zeros(numTrainingMachines^2,n_reduced^2);
        c_sample_avg = zeros(numTrainingMachines^2,1);
        for idxLag = 1:numCov
            idxMatLag = (numTrainingMachines^2)*(idxLag-1) + ...
                (1:(numTrainingMachines^2));
            c_multiplier_avg = c_multiplier_avg + c_multiplier(idxMatLag,:);
            c_sample_avg = c_sample_avg + c_sample(idxMatLag);
        end
        % BETA increases as expected from the cov multiplier and sample
        % increasing due to summation
        BETA_lag = 17;
        % Get objective function for weights
        objFun = @(x) Weights(x,a,c_multiplier_avg,c_sample_avg,BETA_lag);
        % Set inequalities for weights
        Aineq = [-eye(numTrainingMachines); eye(numTrainingMachines)];
        bineq = [zeros(numTrainingMachines,1);ones(numTrainingMachines,1)];
        % Initialization
        w0 = 0.5*ones(numTrainingMachines,1);
        % Nonlinear optimizer options
        opt = optimoptions('fmincon', ...
            'Display','iter-detailed', ...
            'StepTolerance',1e-25,...
            'MaxFunEvals',3e9, ...
            'OptimalityTolerance',1e-12, ...
            'ConstraintTolerance',1e-12);
        % Solve
        weights = fmincon(objFun,w0,Aineq,bineq,[],[],[],[],[],opt);
end

%% Process

% Get learned A
disp( "Is estimated A PSD: " + isPsd(A) )
disp( "Are diagonals (variances) nonneg: " + all(diag(A)>=0))

% Compute learned covariance w.r.t. no lag (T=0)
C0 = (STD_z(:,:,1)*M^(-1/2))*V*(A.*K(:,:,1))*(V.')*(M^(-1/2)*STD_z(:,:,1));
if any(abs(imag(C0))>eps,'all')
    error('Data is complex due to lack of symmetry in Lm')
end

switch optimizerLoss
    case 'L1_MASK'
        % Convert weights to discrete mask
        w = zeros(size(weights));
        w(weights>0.5) = 1;
end

%% Visualize learned covariance/correlation matrix

% Visualize w.r.t. no lag (T=0)

% Setup title
lossFuncChar = strrep([' (' char(optimizerLoss) ')'],'_','');

% Calculate covariance sample (without noise)
C0_sample = STD_z0(:,:,1)*(z.')*z*STD_z0(:,:,1) / size(z,1);
% Calculate covariance sample (WITH noise)
C0_sample_noisy = ones(n,n)*NaN;
C0_sample_noisy(trainingMachines,trainingMachines) = trainingC_sample_norm(:,:,1);
% Calulate errors
C0_sample_MINUS_C0 = {abs(C0_sample-C0)};
C0_sample_noisy_MINUS_C0 = {trainingMachines,trainingMachines, ...
    abs(C0_sample_noisy(trainingMachines,trainingMachines) - ...
    C0(trainingMachines,trainingMachines))};
C0_sample_noisy_MINUS_C0_sample = {trainingMachines,trainingMachines, ...
    abs(C0_sample_noisy(trainingMachines,trainingMachines) - ...
    C0_sample(trainingMachines,trainingMachines))};
C0diff_all0 = [C0_sample_MINUS_C0{1}(:); C0_sample_noisy_MINUS_C0{3}(:); ...
    C0_sample_noisy_MINUS_C0_sample{3}(:)];
C0diff_all0(isnan(C0diff_all0)) = mean(C0diff_all0(~isnan(C0diff_all0)), ...
    'all');

% Heuristics for color scale for cov/corr matrices
C0_all = [C0 C0_sample C0_sample_noisy];
colorLimits = [min(C0_all,[],'all') max(C0_all,[],'all')];
% Plot covariances
H_COV = figure;
if optimizerNormalize
    colorLimits(1) = 0;
    % Learned from A
    subplot(2,2,1)
    heatmap(abs(C0),'ColorLimits',colorLimits)
    title('|Estimated|')
    % Expected (clean)
    subplot(2,2,3)
    heatmap(abs(C0_sample),'ColorLimits',colorLimits)
    title('|Expected| (clean sample)')
    % Sample (noisy)
    subplot(2,2,2)
    heatmap(trainingMachines,trainingMachines,abs(C0_sample_noisy( ...
        trainingMachines,trainingMachines)),'ColorLimits',colorLimits)
    title('|Expected| (noisy sample)')
else
    % Learned from A
    subplot(2,2,1)
    heatmap(C0,'ColorLimits',colorLimits)
    title('Estimated')
    % Expected (clean)
    subplot(2,2,3)
    heatmap(C0_sample,'ColorLimits',colorLimits)
    title('Expected (clean sample)')
    % Sample (noisy)
    subplot(2,2,2)
    heatmap(trainingMachines,trainingMachines,C0_sample_noisy( ...
        trainingMachines,trainingMachines)),title('Expected (noisy sample)')
end
sgtitle(['Covariance of w(t)' lossFuncChar])
% Plot weights if relevant
if strcmpi(optimizerLoss,'L1_MASK')
    % Split machines between good and bad
    idxGood = ~ismember(trainingMachines,BAD_MACHINES);
    machinesGood = trainingMachines(idxGood);
    machinesBad = trainingMachines(~idxGood);
    wGood = weights(idxGood);
    wBad = weights(~idxGood);
    % Plot
    subplot(2,2,4)
    stem(machinesGood(:),wGood(:))
    hold on
    stem(machinesBad(:),wBad(:))
    title('Weights')
end

% Heuristics for color scale for error of cov/corr matrices
if includeRandomLargeNoise
    % If one or more machines is extremely noisy, the variance of that
    % one machine will be extremely higih and mess up with the scale of
    % the remaining data points
    C0diff_all = rmoutliers(C0diff_all0);
    if isscalar(unique(C0diff_all))
        C0diff_all = C0diff_all0;
    end
else
    C0diff_all = C0diff_all0;
end
colorLimits = [min(C0diff_all) max(C0diff_all)];
% Plot absolute error
figure
% Should be zero
subplot(2,2,3)
heatmap(C0_sample_MINUS_C0{:},'ColorLimits',colorLimits)
title('|Estimated - Clean sample|')
% Should have sparse errors
subplot(2,2,2)
heatmap(C0_sample_noisy_MINUS_C0{:},'ColorLimits',colorLimits)
title('|Estimated - Noisy sample|')
% Should have sparse errors
subplot(2,2,1)
heatmap(C0_sample_noisy_MINUS_C0_sample{:},'ColorLimits',colorLimits)
title('|Noisy sample - Clean sample|')
% Title
sgtitle(['Error of covariance of w(t)' lossFuncChar])
