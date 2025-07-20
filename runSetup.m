%% Reproducibility
% Set random seed for reproducibility

% Default
if ~exist('seed','var')
    seed = 1;
end

% Set
rng(seed)

%% System
% Set up bus system using MATPOWER
% Default
if ~exist('fileName','var')
    fileName = 'case30.m';
end
% Set
[~,~,ext] = fileparts(fileName);
if strcmpi(ext,'.raw')
    mpc = psse2mpc(fileName);
else
    mpc = loadcase(fileName);
end

% Size of system (i.e. number of generators)
n = size(mpc.gen,1);

% Calculate system characteristics
% If missing system information, randomly generate parameter values
[L,M,D, MPC,  P_load,Q_load,busesInclude,busesExclude]  ...
    = calc_L_M_D(mpc,n,fileName);

%% Simulation time
% Define simulation time
deltaT = 1e-3;
t0 = (0:deltaT:10).';

%% Frequency-based model reduction

% For inter-area oscillations, the frequency range can described as a
% two-valued vector specifyng lower and upper frequencies of interest e.g.:
% freqs = [f_lower f_upper]

% Set desired frequency range for reduction of eigenspace
freqs = []; % Empty = no model reduction

% Set desired frequency range for filtering of inputs
% NOTE: Typically, wnat freqsFilt and freqs to be the same for learning
inputFilter = false;
freqsFilt = [0.2 0.8];


%% Generate data
% Data generation
% p (Input covariance)
COV_P = 0.01*M^2;

%% Noise
% Define noise, noting that if the inclusion is set to false, the
% subsequent variables for that noise type will not be used in simulation

% NOTE: The "_Magnitude" and "_Var" at the end of noise parameter variables
% denote magnitude and variance of the randomly generated noise.

% Gaussian noise (trivial noise)
includeNoise = false;
varianceNoise_per_second = rad2deg(0.01);

% Random large errors
includeRandomLargeNoise = false;
randomLargeNoise_MachinesAffected = 5;
randomLargeNoise_Magnitude = 1; % Typically +/- 1 deg noise on angles
randomLargeNoise_Var = 0;
randomLargeNoise_NumInjections = 3;

% False data injection
includeFalseInjection = false;
falseInjection_MachinesAffected = 5;
% Falsify network (load/branch) parameters by randomly perturbing values 
% in MPC
falseInjection_G_perturbation_mult = 1;
falseInjection_G_perturbation_var = 0;
falseInjection_B_perturbation_mult = 1.5;
falseInjection_B_perturbation_var = 0.1;
falseInjection_GB_percentLoads = 0.5;
% Falsify genreator system parameters by randomly perturbing M,L,D
falseInjection_systemPerturbation = 0.1;
% How power mismatch is perturbed
% Options:
% 'same'          - Use same p(t) as true system
% 'diff_Gaussian' - Random generation of p(t) is a Gaussian distribution
%                   different from that of the true system
falseInjection_p = 'same';

% PMU Drift (synchronization error)
includeDriftNoise = false;
driftNoise_MachinesAffected = 5;
driftNoise_deltaT = 10*deltaT; % Drift time per second
driftNoise_pps = 1; % PPS = pulse per second, used to sync to UTC
    
%% Learning

% Create training and test data split for machines
testMachines = 1;
trainingMachines = setdiff(1:n,testMachines);

% Time points
% 1. Training times correspond to time ticks of the data that are used to
%    learn 'A' (used to calculate the covariance matrix in MOM optimization)
trainingTimesIdx = (1:round((1/deltaT)/60):numel(t0)).';
% 2. Prediction and test times correspond to time ticks that are used to
%    predict unmetered machines. If we want to find E[x2|x1] where x2 is
%    the unmetered buses and x1 are the metered buses, then prediction
%    times are te time ticks for x1 and test times are the time ticks for
%    x2
predictionTimesIdx = trainingTimesIdx;
testTimesIdx = trainingTimesIdx;

% Optimization parameters
% Loss function options:
% 'L2'      || C0_sample - C0_est   ||2
% 'L1'      || C0_sample - C0_est   ||1
% 'L1_MASK' ||(C0_sample - C0_est)*M||2 + GETA*||W||2
optimizerLoss = 'L2';
% Enforce learned A to be symmetrical?
optimizerASymm = true;
% Normalize data standard deviation? Use correlation in lieu of covariance?
optimizerNormalize = false;

% Would you like to only predict from buses with non-noisy (i.e. good)
% machines?
% Explanation:
% - When we estimate signal of unmetered buses x2 using metered buses x1,
%   x1 may include data streams from noisy (bad) machines.
% - If you set "useOnGoodMacines" to true, then during prediction, the
%   training machines with non-trivial noise on them will be excluded
% - In other words, x1 will be pruned during the prediction process (but
%   NOT during the learning of A in the MOM-based optimization)
useOnGoodMachines = false;

%% Predictions

% Matrix inversion
% When calculating inverse of C11, you can simply compute the inverse or
% use the woodbury matrix identity.
% PRO of using identity: Better for conditioning
% CON of using identity: Must add noise a diagonal Gaussian noise matrix
useWoodburyMatIdentityForInverse = false;