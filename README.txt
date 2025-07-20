
This is the repository for enhancing a Gaussian Process (GP) for power 
system state estimation to be robust to noisy and/false data. The
state-of-the-art Gaussian Process that we are enhancing can be found in this
paper:
[1] M. Jalali,V. Kekatos,H. Zhu, and V.A. Bhela, Siddharth and Centeno,
"Inferring power system dynamics from synchrophasor data using gaussian
processes," IEEE Transactions on Power Systems, vol.37, no.6,pp. 4409–4423,
2022.

How to use:
You can reproduce results from paper by running scripts with
specific cases set up:
- "case_falseData.m"
- "case_largeNoise.m"
- "case_drift.m"
- "case_largeSys.m"
To see a simplest run with just the skeleton, using the classical L2 loss
function, run the script:
- "case_tryMe.m"
Alteratively, you can customize your own workflow in the following steps:
   1. Set up state estimation parameters
   - You can directly run script "runSetup.m"
   - Optionally, you can define pre-setup variables that define the setup
     - 'seed' is the random number generator seed for reproducibility. If
        you do not define it before setup, it will default to 1.
     - 'fileName' is the .raw of .m file specifying the power system of
       interest. IF you do not define it before setup, the default is
       IEEE 30 bus sytem.
   2. (Optional) Manually modify the default setup parameters
   - You can manually modify the variables that were defined when you ran
     "runSetup.m" in step (1).
   - The best way to understand which variables you would like modify is
     to refer to the setup script itself, which is well commented, but
     below is a non-comprehensive summary of variables you can consider
     adjusting for your purposes
     DATA GENERATION (SIMULATION OF LINEAR SYSTEM)
     - 'deltaT','t0' are the time step and time vector for simulating the
       linearized system for data generation
     - 'COV_P' sets covariance matrix for input power p(t) for data
        generation
     DATA PROCESSING/NOISE INCLUSION
     - 'inputFilter','freqsFilt','freqs' adjust the frequency ranges of
        interest for model reduction
     - 'includeNoise','includeRandomLargeNoise','includeFalseInjection',
       'includeDriftNoise' are logical scalars for determining whether the
        noise type should be included. They have corresponding variables
        that specify in more detail the characteristics of the noise.
     LEARNING/PREDICTION
     - 'trainingMachines','testMachines' are vectors of machines with 
       measurements (buses with meters) and machiens without measurements
       (buses that are un-metered)
     - 'trainingTimesIdx' are indices corresponding to time ticks in the
       simulation time 't0' that are actually used to learn 'A' (used to
       calculate the covariance matrix in MOM optimization)
     - 'optimizerLoss' is the loss function in the learning 'A' during
        MOM optimization
     - 'predictionTimesIdx','testTimesIdx' are indicies corresponding to
       time ticks in 't0' that are used to predict. If we want to find
       E[x2|x1] where x2 is the unmetered buses and x1 are the metered
       buses, then prediction times are te time ticks for x1 and test times
       are the time ticks for x2
     - 'optimizerNormalize' is a logical dictating whether to use
        normalized data (correlation matrices) over the original un-
        normalized data (covariance matrices)
      - 'useWoodburyMatIdentityForInverse' is a logical dictating whether
        to use a matrix inversion lemma for better conditioning when
        solving the inverse of a large covariance matrix
   3. Run "runGenerateData.m"
   4. Run "runLearn.m"
   5. Run "runVisualizeTrainingResults"

Other software needed:
- Requires MATPOWER to create system parameters
- Need cvx for linear optimization

Data files:
- All .m files come from MATPOWER
- The exception: The IEEE 300 bus case file is taken from the Github
  referenced in the paper [1] because it contains generator information;
  Github is found:
  https://github.com/manajalali/GP4GridDynamics