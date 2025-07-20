%% Covariance/correlation matrices

% Set up for using covariance or correlation
if optimizerNormalize
    % Find standard deviation 
    % ...for NOISY training data
    std_ztraining = std(trainingZ);
    % ...for test data
    std_ztest = std(z(:,testMachines));
    % ... for test and CLEAN training data
    std_z0 = std(z);

    % Get standard deviation multiplier
    % ...for noisy training data
    STD_ztraining = diag(1./std_ztraining);
    % ...for test and noisy training data combined
    std_z = zeros(1,n);
    std_z(trainingMachines) = std_ztraining;
    std_z(testMachines) = std_ztest;
    STD_z = diag(1./std_z);
    % ...for test and CLEAN training data
    STD_z0 = diag(1./std_z0);

    % Sanity check - if no noise, then noisy and clean standard deviations
    % should be identical
    if isempty(BAD_MACHINES)
        relTol_trainStdDev = abs((std_z0(trainingMachines)-std_ztraining) ...
            ./std_ztraining);
        assert(all(relTol_trainStdDev<0.001))
    end
else
    % Set all standard deviation matrices to identity
    STD_ztraining = eye(numTrainingMachines);
    STD_z = eye(n);
    STD_z0 = eye(n);
end

% Get parameterized covariance/correlation vector
% - Want to get parameterized covariance (or correlation?) matrix in a
%   vectorized format for easy learning of 'A'
% - According to paper ("Inferring Power System Frequency Oscillations
%   using Gaussian Processes" by Jalali et. al.), the vectorized covariance
%   multiplier is:
%      CT = U*vec(A dot KT) = U*diag(kt)*a
%   such that:
%       kt := vec(KT)
%       U  := (S_M*M^(-1/2)*V) (x) (S_M*M^(-1/2)*V)
% - If we are not using speed omega as training variable but using p(t),
%   then M^(-1/2) will be replaced by M^(1/2)
kt = KT(:);
U = kron(STD_ztraining*S_M*(M^(-1/2))*V,STD_ztraining*S_M*(M^(-1/2))*V);
c0_multiplier = U*diag(kt);

% Get measured covariance/correlation vector
trainingC0_sample_norm = STD_ztraining*trainingC0_sample*STD_ztraining;
if includeNoise
    % Subtract trivial Gaussian additive noise
    trainingC0_sample_norm = trainingC0_sample_norm + ...
        eye(numTrainingMachines)*sigmaNoise^2;
end
c0_sample = trainingC0_sample_norm(:);

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
        % Get objective function for weights
        objFun = @(x) Weights(x,a,c0_multiplier,c0_sample,BETA);
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

% Compute learned covariance
C0 = (STD_z*M^(-1/2))*V*(A.*KT)*(V.')*(M^(-1/2)*STD_z);
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


% Setup title
lossFuncChar = strrep([' (' char(optimizerLoss) ')'],'_','');

% Calculate covariance sample (without noise)
C0_sample = STD_z0*(z.')*z*STD_z0 / size(z,1);
% Calculate covariance sample (WITH noise)
C0_sample_noisy = ones(n,n)*NaN;
C0_sample_noisy(trainingMachines,trainingMachines) = trainingC0_sample_norm;
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
