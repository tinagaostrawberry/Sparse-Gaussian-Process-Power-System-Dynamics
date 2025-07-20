%% Set up state estimation parameters

% Setup
% NOTE: The seed and system parameters are important here because the
% clarity of the interpretation of each steps during visualization is
% dependent on how strongly correlated each timestep is
seed = 1;
fileName = 'case30.m';
runSetup

%% Modify the default setup parameters

% Training time
trainingTimesIdx = (1:round((1/deltaT)/240):numel(t0)).';
% Select prediction/test time to be from 5 to 7.5 seconds, with slightly
% lower sampling rate for faster computation
[~,idxStart] = min(abs(t0(trainingTimesIdx)-5));
[~,idxEnd] = min(abs(t0(trainingTimesIdx)-7.5)); 
predictionTimesIdx = trainingTimesIdx(idxStart:2:idxEnd);
testTimesIdx = trainingTimesIdx(idxStart:2:idxEnd);

% Set inverse calculation (which requires noise)
useWoodburyMatIdentityForInverse = true;
varianceNoise_per_second = 1e-8;
includeNoise = true;

% Set learning parameters
optimizerNormalize = true;
BETA = 0.8;

% Generate train/noise/test machine split
testMachines = 1;
trainingMachines = setdiff(1:n,testMachines);
noisyMachines = 4;

% Random large errors
% Set up for the generation of data with random large errors
includeRandomLargeNoise = true;
randomLargeNoise_MachinesAffected = noisyMachines;
% Set parameters of noise
randomLargeNoise_NumInjections = 21;
randomLargeNoise_Magnitude = 0.1;

%% Generate data
runGenerateData

%% Learn and perform prediction

% Try each optimizer
optimizerTypeList = ["L1_MASK","L2"];
for optimizerTypeIdx = 1:numel(optimizerTypeList)
    optimizerLoss = optimizerTypeList(optimizerTypeIdx);
    
    % Learn
    runLearn

    % Predict
    useOnGoodMachines = false;
    runVisualizeTrainingResults

    % Format prediction
    f = gcf;
    rmseVal = getFigureTitleInfo(f,lossFuncChar);
    formatFigures(f,rmseVal,[],strcmpi(optimizerLoss,"L2"))

    % Interpret steps
    switch optimizerLoss
        case "L1_MASK"
            % To understand significance of step 2 (i.e. the masking), use
            % the L1-learned A without the mask

            % Unlearn mask and predict
            useOnGoodMachines = false;
            w = 0*w;
            runVisualizeTrainingResults

            % Format plot
            f = gcf;
            rmseVal = getFigureTitleInfo(f,lossFuncChar);
            formatFigures(f,rmseVal,[],strcmpi(optimizerLoss,"L2"))
            title("GP Prediction with L1 loss (RMSE: " + num2str(rmseVal) + ")")

        case "L2"
            % To understand significance of step 1 (i.e. the sparse
            % learning of A), use the L2-learned A but with ideal weights

            % Emulate learning weights by only predicting on good machines
            useOnGoodMachines = true;
            runVisualizeTrainingResults

            % Format plot
            f = gcf;
            rmseVal = getFigureTitleInfo(f,lossFuncChar);
            formatFigures(f,rmseVal,[],strcmpi(optimizerLoss,"L2"))
            title(" GP Prediction with Mask (RMSE: " + num2str(rmseVal) + ")")

    end

end
