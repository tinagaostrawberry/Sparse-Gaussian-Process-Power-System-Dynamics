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
numPredMachines = numel(predictionMachines);
if usePredMachines
    predictingZ = trainingZ(ismember(trainingTimesIdx,predictionTimesIdx), ...
        ismember(trainingMachines,predictionMachines));
else
    predictingZ = trainingZ(ismember(trainingTimesIdx,predictionTimesIdx),:);
end

% Get machines used for test
S_test = I_N(testMachines,:);
sqrtM = M^(-1/2);

%% Prepare sparsification of C11

% Get covariance matrix of ALL testing machines
C_all = (STD_z0*sqrtM*V) * ( A .* KT ) * ((V.')*sqrtM*STD_z0);
% Perform  clustering
rng(0) % Reset random number gen for reproducibility of kmedoids
[idxK,C,sumd,distK] = kmedoids(C_all,numK);

% "Training machine contribution" is a logical matrix where the [i,j]th
% entry denotes whether the jth prediction machine will contribute to
% the prediction of the ith test machine
% - When the matrix is all true, no sparsification occurs and the
%   inverse is taken of the entire C11 matrix
% - When the matrix has some false entries, the C11 matrix is
%   sparsified, resulting in the rows and columns of the C11 matrix
%   being removed, and the inverse if taken of the reduced matrix
trainingMachContribut = true(numTestMachines,numPredMachines);
for testMachIdxContribut = 1:numTestMachines
    % Find other machines in the same group
    testMachContribut = testMachines(testMachIdxContribut);
    clusterIdx = idxK(testMachContribut);
    contributIdx = ismember(predictionMachines,find(clusterIdx==idxK));
    if sum(contributIdx) < 2
        % If no trainining machines are in same group (or too few),
        % then include any training machines by whose peak contribution
        % passes a fractional threshold of 0.85
        C21_all = (S_test(testMachIdxContribut,:)*STD_z0*sqrtM*V) * ...
            ( A .* KT ) * ...
            ((V.')*sqrtM*STD_z0*(S_pred.'));
        contributThresh = 0.85*max(abs(C21_all));
        contributIdx = abs(C21_all) >= contributThresh;
        trainingMachContribut(testMachIdxContribut,:) = contributIdx;
    else
        % Only use contribution of machines that are in the same
        % group
        trainingMachContribut(testMachIdxContribut,:) = contributIdx;
    end
end

% Set number of test machines that we will perform prediction for
numTestMachContribut = numTestMachines;

% If aggregate representation...
if aggregateRepresentation
    % For aggregrate representation, assume that originally wanted to
    % estimate behavior at ALL buses (i.e. test set is all n machines).
    % Thus, aggregrate representation reduces the number of buses that
    % we need to predict by only predicting the representative buses in
    % each group
    assert(numTestMachines==n)
    % Ignore test machine test and only care about representative
    % machines in each cluster
    numTestMachContribut = numK;
    % Modify training contribution and selection matrices to do the
    % same
    [testMachinesCluster,~] = ind2sub(size(distK),find(distK==0));
    trainingMachContribut = trainingMachContribut(testMachinesCluster,:);
    S_test = S_test(testMachinesCluster,:);

end

%% Covariance matrices for prediction

% TO DO: Clean up code so no need to use temporary variables, which are
% confusing and inefficient

% Create temporary variables as we iterate through each machine we are
% predicting for
% Store predictions
test_est_TEMP = zeros(numTestTimeSamples,numTestMachContribut);
test_err_all_diag_TEMP = zeros(numTestTimeSamples,numTestMachContribut);
% Make temporary copy of original selection matrices
S_pred_TEMP = S_pred;
S_test_TEMP = S_test;
% Make temporary copy of predicting data
predictingZ_TEMP = predictingZ;

% Create variables for analysis
sizesC11 = zeros(numTestMachContribut,1);

% Sanity check - contributions of test machines are the same
assert(size(trainingMachContribut,1)==numTestMachContribut)

tic
for idxContribut = 1:numTestMachContribut
    % In each iteration, only grab the submatrix useful for prediction on
    % this particular test machine
    S_pred = S_pred_TEMP(trainingMachContribut(idxContribut,:),:);
    S_test = S_test_TEMP(idxContribut,:);
    predictingZ = predictingZ_TEMP(:,trainingMachContribut(idxContribut,:));

    % ------------------------------- C11 ---------------------------------
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
    else
        C11 = kron(S_pred*sqrtM*V,eyeT11) * ...
            ( A11 .* K11 ) * ...
            kron((V.')*sqrtM*(S_pred.'),eyeT11);
        C11 = C11 + sigmaNoise^2 *eye(size(C11,1));
    end

    % ------------------------------- C22 ---------------------------------
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

    % ------------------------------- C21 ---------------------------------
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

    % Predict signals   
    if useWoodburyMatIdentityForInverse
        test_est = C21*invC11*predictingZ(:);
        testErr = C22 - C21*invC11*(C21.');

        sizesC11(idxContribut) = size(invC11,1);
    else
        test_est = C21*(C11\predictingZ(:));
        testErr = C22 - C21*(C11\(C21.'));

        sizesC11(idxContribut) = size(C11,1);
    end
    % Save predicted signals in temporary variables
    test_est_TEMP(:,idxContribut) = test_est;
    test_err_all_diag_TEMP(:,idxContribut) = diag(testErr);
end
time = toc;
clear U0 C00 V0 A0

% Clear temporary variables as we revert to old
test_est = test_est_TEMP;
testErr = diag(sparse(test_err_all_diag_TEMP(:))); % Sparse needed to prevent unresponsiveness
S_pred = S_pred_TEMP;
S_test = S_test_TEMP;
predictingZ = predictingZ_TEMP;
%
clear test_est_TEMP test_err_all_diag_TEMP S_pred_TEMP S_test_TEMP

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

if ~aggregateRepresentation
    % Initialize for uncertainty characterization
    yint2_0 = full(diag(testErr_plot));
    rmseList_normalized = zeros(1,size(testZ_plot,2));

    % Iterate through each unmetered bus
    for idxTest = 1:size(testZ_plot,2)
        % Plot
        hfig = figure;
        plot(testTimes-testTimes(1),testZ_plot(:,idxTest),'-k','Linewidth',2.5)
        hold on
        plot(testTimes-testTimes(1),estZ_plot(:,idxTest),'.-b','Linewidth',2.5, ...
            'MarkerSize',15)
        % Prediction intervals for 2 stddev
        machineIdx = (1:numel(testTimes)) + (idxTest-1)*numel(testTimes);
        test_err_machineIdx = abs(yint2_0(machineIdx));
        yint2 = [estZ_plot(:,idxTest) + 2*sqrt(test_err_machineIdx) ...
            estZ_plot(:,idxTest) - 2*sqrt(test_err_machineIdx)];
        patch([testTimes-testTimes(1);flipud(testTimes-testTimes(1))], ...
            [yint2(:,1);flipud(yint2(:,2))],'b','FaceAlpha',0.1);
        % Calculate RMSE
        rmseList_normalized(idxTest) = rmse(estZ_plot(:,idxTest),testZ_plot(:,idxTest)) ...
            / (max(testZ_plot(:,idxTest))-min(testZ_plot(:,idxTest)));
        % Title
        legend(["Expected","Est","95% error"],"ItemHitFcn",@cb_legend)
        title1 = "Machine " + testMachines(idxTest) + string(lossFuncChar);
        title2 = "Min-max-normalized RMSE: " + num2str(rmseList_normalized(idxTest));
        title3 = "Cluster "+idxK(testMachines(idxTest));
        titleStr = [title1 title2 title3];
        title(titleStr)
        % Labels
        xlabel('Time [s]')
        ylabel('Speed [rad/s]')
        % Formatting
        xlim([predTimes(1) predTimes(end)]-testTimes(1))
    end


else
    % If aggregate representation of test machines, overlay the transient
    % response of the test machines
    gradientColors = repmat([0 0 0],numK,1);
    for clusterIdx = 1:numK
        % Overlay expected test machine responses
        hfig = figure;
        linExp = plot(testTimes,testZ_plot(:,clusterIdx==idxK), ...
            'Color',gradientColors(clusterIdx,:),'LineWidth',1.5);
        % Plot aggregrate representative
        hold on
        linAgg = plot(testTimes,estZ_plot(:,clusterIdx),'.-b','Linewidth',2.5);
        % Legend
        legend([linExp(1) linAgg],["Aggregate","Representative Est"])
        % Formatting
        xlabel('Time')
        ylabel('w')
        title(['Cluster ' num2str(clusterIdx)])

    end
end

