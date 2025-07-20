%% Set up state estimation parameters

% Setup
seed = 2;
fileName = 'case1354pegase.m';
runSetup

%% Modify the default setup parameters

% Simulation time
deltaT = deltaT/10;
t0 = (0:deltaT:(275/50)).';
% Training time
trainingTimesIdx = (1:round((1/deltaT)/50):numel(t0)).';
% Select prediction/test time to be from 0 to 5 seconds
[~,idxStart] = min(abs(t0(trainingTimesIdx)-0));
[~,idxEnd] = min(abs(t0(trainingTimesIdx)-5));
predictionTimesIdx = trainingTimesIdx(idxStart:idxEnd);
testTimesIdx = trainingTimesIdx(idxStart:idxEnd);

% Set inverse calculation (which requires noise)
useWoodburyMatIdentityForInverse = true;
varianceNoise_per_second = 1e-5;
includeNoise = true;

% Set filter
inputFilter = true;
freqsFilt = [1 1.75];
freqs = freqsFilt;

% Set learning parameters
optimizerNormalize = true;
BETA = 50;
% Set optimizer
optimizerLoss = 'L1_MASK';

% Randomly generate test/train/noise split
testMachines = 1:n;
rangeTrainingMachs = [80 110];
trainingMachines = randperm(n,randi(rangeTrainingMachs));
rangeBadMachines = [5 10];
noisyMachines = trainingMachines( ...
    randperm(numel(trainingMachines),randi(rangeBadMachines)));

% Bad noise injection
% Set up for the generation of data with bad noise injection
includeFalseInjection = true;
falseInjection_MachinesAffected = noisyMachines;
% Falsify network (load/branch) parameters by randomly perturbing values 
% in MPC
falseInjection_G_perturbation_mult = 5;
falseInjection_G_perturbation_var = 0;
falseInjection_B_perturbation_mult = 1;
falseInjection_B_perturbation_var = 0;
falseInjection_GB_percentLoads = 0.1;
% Falsify generator system parameters by randomly perturbing M,L,D
falseInjection_systemPerturbation = 0;
% How power mismatch is perturbed
falseInjection_p = 'diff_Gaussian';

%% Generate data
runGenerateData

%% Set up variables for saving/simulation

% Save simulation times
timeStruct = struct();
timeStruct.(optimizerLoss) = zeros(1,3);

% Set number of clusters for sparsifying C11
numK = 26;

%% Learn

runLearn
close all

%% Peform large system prediction

% ---------- Sparsifying C11, but not aggregrate representation -----------
aggregateRepresentation = false; %#ok<NASGU>
case_largeSys_visualizeTrainingResults
% Record time
timeStruct.(optimizerLoss)(1) = time;
% Record covariance sizes
sizesC11_maxMin = [max(sizesC11) min(sizesC11)];
save('sizesC11_maxMin','sizesC11_maxMin')
% Record plots of predictions
hFIGSALL = findobj('type','figure');
numFIGSALL = length(hFIGSALL);
LIST = zeros(numFIGSALL,3);
for idxFig = 1:numFIGSALL
    hfig = figure(idxFig);
    savefig(hfig,"fig"+num2str(idxFig)+"_sparsifyC11.fig")
    %
    TIT = get(get(hfig.CurrentAxes,'title'),'string');
    LIST(idxFig,1) = str2double(strrep(TIT{2},'Min-max-normalized RMSE: ',''));
    LIST(idxFig,2) = idxFig;
    LIST(idxFig,3) = str2double(strrep(TIT{3},'Cluster ',''));
end
save('LIST','LIST')
close all


% ------------- Sparsifying C11 AND aggregrate representation -------------
aggregateRepresentation = true;
case_largeSys_visualizeTrainingResults
% Record time
timeStruct.(optimizerLoss)(2) = time;
% Record plots of aggregation
hFIGSALL =  findobj('type','figure');
numFIGSALL = length(hFIGSALL);
for idxFig = (1:numFIGSALL)
    hfig = figure(idxFig);
    savefig(hfig,"fig"+num2str(idxFig)+"_SparseAndAggRep.fig")
end
close all

% -------------------------- Original prediction --------------------------
runVisualizeTrainingResults
% Record time
timeStruct.(optimizerLoss)(3) = time;
close all


% Save time structure
save('timeStruct','timeStruct')

%% Visualize

% Look at times
load('timeStruct')
timeSparse = timeStruct.(optimizerLoss)(1);
disp("Time to simulate with sparsification of C11: " + num2str(timeSparse))
%
timeSparseAndAgg = timeStruct.(optimizerLoss)(2);
disp("Time to simulate with sparsification of C11 and aggregation: " ...
    + num2str(timeSparseAndAgg))
%
timeOrig = timeStruct.(optimizerLoss)(3);
disp("Time to simulate with only original: " + num2str(timeOrig))

% Look at sparsification and aggregate representation results of
% representative RMSEs
% Sort LIST that maps figures to RMSEs
load('LIST')
[~,idxSort] = sort(LIST(:,1));
list_SORT = LIST(idxSort,:);
% Visualize interquartile samples
numFig = size(list_SORT,1);
idxQs = [round(numFig/4) round(numFig/2) round(numFig*3/4)];
for idx = idxQs
    % View results of sparisification of C11
    figNum = list_SORT(idx,2);
    clusterNum = list_SORT(idx,3);
    fig1 = open("Fig"+num2str(figNum)+"_sparsifyC11.fig");
    % Get prediction results
    [rmseVal,mach] = getFigureTitleInfo(fig1,lossFuncChar);
    % Re-format
    formatFigures(fig1,rmseVal,mach)
    ylim([-0.01 0.01])

    % View results of sparisification of C11 and aggregrate representation
    fig2 = open("Fig"+num2str(clusterNum)+"_SparseAndAggRep.fig");
    % Re-format
    formatFigures(fig2,[],mach)
    ylim([-0.01 0.01])
end


