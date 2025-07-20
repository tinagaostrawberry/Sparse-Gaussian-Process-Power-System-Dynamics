function y = Weights(w,a,c0_multiplier,c0_sample,BETA)

% Loss function for learning the weights that mask out bad data streams
%
% Inputs:
% 'w'             - Vector of decision variables representing the weights
%                   corresponding to each machine
% 'a'             - Vectorized matrix A corresponding to covariances of the
%                   eigeninputs
% 'c0_multiplier' - Vector of multipliers for the parameterization of the
%                   learned covariance
% 'c0_sample'     - Vectorized sample covariance matrix of generator
%                   training variable (e.g. speed)
% 'BETA'          - Regularization term for the weights
%
% Outputs:
% 'y'             - Loss value

% Create mask
W = w*(1-w)' + (1-w)*w' + w*w';
M = 1 - W;

% Create main loss term
yCov = sum( abs(c0_multiplier*a - c0_sample) .* M(:) );
% Create L1 term for weights
yW = sum( abs(w) );

% Final loss term
y = yCov + BETA*yW;
end