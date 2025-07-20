%% Set up state estimation parameters

% Setup
seed = 156;
fileName = 'case300.m';
runSetup

%% Modify the default setup parameters

% Simulation time
t0 = (0:deltaT:375).';
% Training time
trainingTimesIdx = (1:round((1/deltaT)/30):numel(t0)).';
% Select prediction/test time to be from 10 to 20 seconds
[~,idxStart] = min(abs(t0(trainingTimesIdx)-10));
[~,idxEnd] = min(abs(t0(trainingTimesIdx)-20));
predictionTimesIdx = trainingTimesIdx(idxStart:idxEnd);
testTimesIdx = trainingTimesIdx(idxStart:idxEnd);

% Set inverse calculation (which requires noise)
useWoodburyMatIdentityForInverse = true;
varianceNoise_per_second = 1e-5;
includeNoise = true;

% Set filter
inputFilter = true;
freqsFilt = [0.5 0.8];
freqs = freqsFilt;

% Set learning parameters
optimizerNormalize = true;
BETA = 6;

% Randomly generate test/train/noise split
testMachines = randperm(n,1); 
rangeTrainingMachs = [15 30];
trainingMachines = randperm(n,randi(rangeTrainingMachs));
trainingMachines(trainingMachines==testMachines) = [];
rangeBadMachines = [5 8];
noisyMachines = trainingMachines( ...
    randperm(numel(trainingMachines),randi(rangeBadMachines)));

% Drift noise
% Set up for the generation of data with drift
includeDriftNoise = true;
driftNoise_MachinesAffected = noisyMachines;
% Set drift time per second
driftNoise_deltaT = deltaT*100;

%% Generate data
runGenerateData

%% Learn and perform prediction

% Try each optimizer
% NOTE: Noise is small, so A learned by L1 and L2 similar; masking is
% critical
optimizerTypeList = ["L1_MASK","L2"];
for optimizerTypeIdx = 1:numel(optimizerTypeList)
    optimizerLoss = optimizerTypeList(optimizerTypeIdx);
    
    % Learn
    switch optimizerLoss
        case "L1_MASK"
            % Learn with respect to larger set of lags for enhanced
            % detection
            numLagsPerSec = 7;
            case_drift_learn
        case "L2"
            runLearn
    end

    % Predict
    runVisualizeTrainingResults

    % Format prediction plot
    fPred = gcf;
    rmseVal = getFigureTitleInfo(fPred,lossFuncChar);
    formatFigures(fPred,rmseVal,[],strcmpi(optimizerLoss,"L2"))

    % Make copy of prediction plot
    fZoom = copyobj(fPred,0);
    % Zoom in on drift error
    xlim([4.6 5.4])
    ylim([0 4]*1e-4/sampleCoeff)
    switch optimizerLoss
        case "L1_MASK"
            title("Zoomed on Proposed Robust GP Prediction")
        case "L2"
            title("Zoomed on Non-robust GP Prediction")
    end

end

