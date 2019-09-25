function [X, Err] = KOMP(A, K_dd, K_yd, T0, K_yy)
% Input: 
%   [A] = column dictionary coefficients
%   [D] = base of the kernel dictionary (same as Y for some cases)
%   [T0] = sparsity level
%   K_dd, K_yd = kernel matrices <D,D> and <Y,D>, respectively
%   N, L =  number of data samples,
%           number of dictionary atoms, respectively
% Output = X that minimize || p(Y) - p(D).A.X||_2 subject to |Xi|_0 <= T0

% Note that the dictionary has to be normalized on the feature space: 
%    i.e.  diagonal elements of "A'*K_dd*A" are 1

X = omp(A'*K_yd',A'*K_dd*A,T0) ; 

Err = [];
if (exist('K_yy','var'))
    Err = zeros(length(K_yy),1);
    for ii=1:length(K_yy)
        Err(ii) = sqrt(K_yy(ii,ii) - 2*K_yd(ii,:)*A*X(:,ii) + X(:,ii)'*A'*K_dd*A*X(:,ii)) ; 
    end
end

end 