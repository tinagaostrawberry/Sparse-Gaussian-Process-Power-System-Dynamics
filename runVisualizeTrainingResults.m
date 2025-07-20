%% Preprocess

% Get machines used for prediction
useMask = strcmpi(optimizerLoss,'L1_MASK');
if useOnGoodMachines && useMask
    usePredMachines = true;
    %
    predictionMachines0 = setdiff(trainingMachines, ...
        [BAD_MACHINES trainingMachines(logical(w))]);
    predictionMachines = trainingMachines(ismember(trainingMachines, ...
        predictionMachines0));
    %
    S_pred = S_M(ismember(trainingMachines,predictionMachines),:);
elseif useOnGoodMachines
    usePredMachines = true;
    %
    predictionMachines0 = setdiff(trainingMachines,BAD_MACHINES);
    predictionMachines = trainingMachines(ismember(trainingMachines, ...
        predictionMachines0));
    %
    S_pred = S_M(ismember(trainingMachines,predictionMachines),:);
elseif useMask
    usePredMachines = true;
    %
    predictionMachines0 = setdiff(trainingMachines,trainingMachines( ...
        logical(w)));
    predictionMachines = trainingMachines(ismember(trainingMachines, ...
        predictionMachines0));
    %
    S_pred = S_M(ismember(trainingMachines,predictionMachines),:);
else
    usePredMachines = false;
    predictionMachines = trainingMachines;
    S_pred = S_M;
end
if usePredMachines
    predictingZ = trainingZ(ismember(trainingTimesIdx,predictionTimesIdx), ...
        ismember(trainingMachines,predictionMachines));
else
    predictingZ = trainingZ(ismember(trainingTimesIdx,predictionTimesIdx),:);
end

% Get machines used for test
S_test = I_N(testMachines,:);
sqrtM = M^(-1/2);

%% Covariance matrices for prediction

%    |    |     |
%    |z_measured|
%    |    |     |      | | |   | C11 | C21^T |
%    | -------- | ~ N( | 0 | , | ----------- | )
%    |    |     |      | | |   | C21 |  C22  |
%    |   z_est  |
%    |    |     |
%
% Such that:
%    C = ( [(M^(-1/2)*V) (x) I_T] * ( A dot KT ) * [ (V^T*M^(-1/2)) (x) I_T])
% where:
%    KT = M^(-1/2)

tic
% ---------------------------------- C11 ----------------------------------
% Get parameters
T11 = predTimes - predTimes.';
K11 = getKernel(lambda,T11,gamma);
%
oneT11 = ones(numPredTimeSamples,numPredTimeSamples);
A11 = kron(A,oneT11);
% Calculate covariance
eyeT11 = eye(numPredTimeSamples);
if useWoodburyMatIdentityForInverse
    U0 = kron(S_pred*sqrtM*V,eyeT11);
    C00 = ( A11 .* K11 );
    V0 = kron((V.')*sqrtM*(S_pred.'),eyeT11);
    A0 = sigmaNoise.^2 *eye(size(U0,1));
    invA = diag(1./diag(A0));
    invC = inv(C00);
    invA_U = A0\U0;
    Chunk = invC + V0*invA_U;
    V_invA = V0/A0;
    invChunkEnd = Chunk\V_invA;
    invC11 = invA-invA_U*invChunkEnd;

    clear U0 C00 V0 A0
else
    C11 = kron(S_pred*sqrtM*V,eyeT11) * ...
        ( A11 .* K11 ) * ...
        kron((V.')*sqrtM*(S_pred.'),eyeT11);
    C11 = C11 + sigmaNoise^2 *eye(size(C11,1));
end

% ---------------------------------- C22 ----------------------------------
% Get parameters
T22 = testTimes - testTimes.';
K22 = getKernel(lambda,T22,gamma);
%
oneT22 = ones(numTestTimeSamples,numTestTimeSamples);
A22 = kron(A,oneT22);
% Calculate covariance
eyeT22 = eye(numTestTimeSamples);
C22 = kron(S_test*sqrtM*V,eyeT22) * ...
    ( A22 .* K22 ) * ...
    kron((V.')*sqrtM*(S_test.'),eyeT22);

% ---------------------------------- C21 ----------------------------------
% Get parameters
T21 = testTimes - predTimes.';
K21 = getKernel(lambda,T21,gamma);
%
oneT21 = ones(numTestTimeSamples,numPredTimeSamples);
A21 = kron(A,oneT21);
% Calculate covariance
C21 = kron(S_test*sqrtM*V,eyeT22) * ...
    ( A21 .* K21 ) * ...
    kron((V.')*sqrtM*(S_pred.'),eyeT11);

% Estimate signals
if useWoodburyMatIdentityForInverse
    test_est = C21*invC11*predictingZ(:);
    testErr = C22 - C21*invC11*(C21.');
else
    test_est = C21*(C11\predictingZ(:));
    testErr = C22 - C21*(C11\(C21.'));
end
time = toc;

% Process predicted signals
estZ = reshape(test_est,numTestTimeSamples,[]);

%% Sampling magnitude

% The covariance parameterization K(.) is derived from taking the
% convolution of the continuous system, but the simulation of the system
% occurs discretely in timesteps. Therefore, there is a magnitude scaling
% based on the sampling rate (this is analogous to the discrete time and
% continuous fourier transforms scaling differently)

% Get scaling assuming sampling rate is consistent
sampleCoeff = deltaT*uniquetol(diff(trainingTimesIdx));
assert(isscalar(sampleCoeff))
testZ_plot = testZ / sampleCoeff;
estZ_plot = estZ / sampleCoeff;
testErr_plot = testErr / sampleCoeff;


%% Visualize predictions and plot uncertainties

% Initialize for uncertainty characterization
yint2_0 = full(diag(testErr_plot));
rmseList_normalized = zeros(1,size(testZ_plot,2));

% Iterate through each unmetered bus
for idx = 1:size(testZ_plot,2)
    % Plot
    hfig = figure;
    plot(testTimes-testTimes(1),testZ_plot(:,idx),'-k','Linewidth',2.5)
    hold on
    plot(testTimes-testTimes(1),estZ_plot(:,idx),'.-b','Linewidth',2.5,'MarkerSize',15)
    % Prediction intervals for 2 stddev
    machineIdx = (1:numel(testTimes)) + (idx-1)*numel(testTimes);
    test_err_machineIdx = abs(yint2_0(machineIdx));
    yint2 = [estZ_plot(:,idx) + 2*sqrt(test_err_machineIdx) ...
        estZ_plot(:,idx) - 2*sqrt(test_err_machineIdx)];
    patch([testTimes-testTimes(1);flipud(testTimes-testTimes(1))], ...
        [yint2(:,1);flipud(yint2(:,2))],'b','FaceAlpha',0.1);
    % Calculate RMSE
    rmseList_normalized(idx) = rmse(estZ_plot(:,idx),testZ_plot(:,idx)) ...
        / (max(testZ_plot(:,idx))-min(testZ_plot(:,idx)));
    % Title
    legend(["Expected","Est","95% error"],"ItemHitFcn",@cb_legend)
    title1 = "Machine " + num2str(testMachines(idx)) + string(lossFuncChar);
    title2 = "Min-max-normalized RMSE: " + num2str(rmseList_normalized(idx));
    titleStr = [title1 title2];
    title(titleStr)
    % Labels
    xlabel('Time [s]')
    ylabel('Speed [rad/s]')
    % Formatting
    xlim([predTimes(1) predTimes(end)]-testTimes(1))
end
