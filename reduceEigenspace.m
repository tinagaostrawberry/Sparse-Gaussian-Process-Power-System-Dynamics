function [lambdaReduced,VReduced,idx] = reduceEigenspace(lambda,V,gamma,freqRange)
% Reduces eigenspace through frequency selectivity
%
% Inputs:
% 'lambda'        - Eigenvalues of the matrix L_M := M^(-1/2)*L*M^(-1/2)
% 'V'             - Eigenvectors of the matrix L_M := M^(-1/2)*L*M^(-1/2)
% 'gamma'         - Ratio beteen inertia M and damping D, is scalar due to
%                   uniform damping assumption
% 'freqRange'     - Range of frequencies of interest
% 
% Outputs:
% 'lambdaReduced' - Reduced set of eigenvalues
% 'VReduced'      - Reduced set eigenvectors
% 'idx'           - Indices of frequencies that are within frequency range

if isempty(freqRange)
    lambdaReduced = lambda;
    VReduced = V;
else
    lambda = lambda(:);

    % Get bandwidths for each eigenspace
    bw = gamma;
    lowerFreq = ( (-gamma+sqrt(gamma^2+4*lambda))/2 - bw/2 )/(2*pi);
    upperFreq = ( ( gamma+sqrt(gamma^2+4*lambda))/2 + bw/2 )/(2*pi);

    % Eliminate eigenspaces that are not within freqRange
    idx = ( (lowerFreq>=freqRange(1)) & (lowerFreq<=freqRange(2)) ) ...
        | ( (upperFreq<=freqRange(2)) & (upperFreq>=freqRange(1))) ...
        | ( (lowerFreq<=freqRange(1)) & (upperFreq>=freqRange(2)));
    lambdaReduced = lambda(idx);
    VReduced = V(:,idx);
    
    disp("Reduced size of Eigenspace: " + num2str(size(VReduced,2)))
end

end



