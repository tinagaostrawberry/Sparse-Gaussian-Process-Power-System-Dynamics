function K = getKernel(lambda,T,gamma)

% Calcualtes the kernel that parameterizes the custom spatio-temporal 
% covariance matrix of eigenstates y' in the paper "Inferring Power System
% Frequency Oscillations using Gaussian Processes" by Jalali et. al. [1]
%
% Inputs:
% 'lambda' - Eigenvalues of the matrix L_M := M^(-1/2)*L*M^(-1/2) and 
%            and are critical in defining the impulse response of the
%            decoupled SISO system as modified from the original MIMO
%            system
% 'T'      - Matrix of time points corresponding the variables of the
%            covariance matrix
% 'gamma'  - Ratio beteen inertia M and damping D, is scalar due to
%            uniform damping assumption
%
% Outpus:
% 'K'      - Kernel matrix

lambda = lambda(:);

% Pre-calculate constants
numEigenstates = length(lambda);
onesNum = ones(numEigenstates);
onesT = ones(size(T));

% Calculate c and d
ci = repmat( -gamma/2 + sqrt(gamma^2-4*lambda)/2, [1 numEigenstates]);
di = repmat( -gamma/2 - sqrt(gamma^2-4*lambda)/2, [1 numEigenstates]);
cj = ci.';
dj = di.';

% Calculate a and b
aij = (-ci.^2) ./ ( (ci+cj).*(ci+dj).*(ci-di) );
bij = ( di.^2) ./ ( (cj+di).*(di+dj).*(ci-di) );
%
aji = (-cj.^2) ./ ( (cj+ci).*(cj+di).*(cj-dj) );
bji = ( dj.^2) ./ ( (ci+dj).*(dj+di).*(cj-dj) );

% Finalize kernel
kij  =  ...
    kron(aij,onesT).*exp(kron(ci, T)) + ...
    kron(bij,onesT).*exp(kron(di, T));
kji_ = ...
    kron(aji,onesT).*exp(kron(cj,-T)) + ...
    kron(bji,onesT).*exp(kron(dj,-T));
K = ...
    kij .*kron(onesNum,heaviside( T)) + ...
    kji_.*kron(onesNum,heaviside(-T));

end
