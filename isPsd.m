function isPsd = isPsd(L)
% Checks if matrix L is positive semidefinite (PSD)
%
% Input:
% 'L'     - Square input matrix
%
% Output:
% 'isPsd' - Logical dictating whether 'L' is PSD

isSym = isSymmetric(L);
eigNonneg = all(eig(L) > -1e-12);
eigReal = all( abs(imag(eig(L))) < 1e-12 );
isPsd = isSym & eigNonneg & eigReal;

end

