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
% Select prediction/test time to be from 180 to 190 seconds
[~,idxStart] = min(abs(t0(trainingTimesIdx)-180));
[~,idxEnd] = min(abs(t0(trainingTimesIdx)-190)); 
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

% Bad noise injection
% Set up for the generation of data with bad noise injection
includeFalseInjection = true;
falseInjection_MachinesAffected = noisyMachines;
% Falsify network (load/branch) parameters by randomly perturbing values
% in MPC
falseInjection_G_perturbation_mult = 1;
falseInjection_G_perturbation_var = 0.5;
falseInjection_B_perturbation_mult = 2;
falseInjection_B_perturbation_var = 0.5;
falseInjection_GB_percentLoads = 1;
% Falsify generator system parameters by randomly perturbing M,L,D
falseInjection_systemPerturbation = 0.05;
% How power mismatch is perturbed
falseInjection_p = 'diff_Gaussian';

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
    runVisualizeTrainingResults

    % Format prediction
    f = gcf;
    rmseVal = getFigureTitleInfo(f,lossFuncChar);
    formatFigures(f,rmseVal,[],strcmpi(optimizerLoss,"L2"))

    % Visualize more clearly the correlation matrix results
    figure
    heatmap(C0_sample_noisy_MINUS_C0{:},'ColorLimits',0.8*colorLimits)
    xlabel('Metered machines')
    ylabel('Metered machines')
    fontsize(15,"points")
    switch optimizerLoss
        case "L1_MASK"
            title('Absolute Error (L1 Loss)')
        case "L2"
            title('Absolute Error (L2 Loss)')
    end  
end
