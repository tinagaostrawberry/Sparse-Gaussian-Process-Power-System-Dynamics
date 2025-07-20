function isSym = isSymmetric(L)
% Evaluates symmetry with a tolerance
%
% Input:
% 'L'     - Input matrix
%
% Output:
% 'isSym' - Logical as to whether 'L' is symmetric
isSym = all( abs(L - L.')<1e-8, 'all' );
end