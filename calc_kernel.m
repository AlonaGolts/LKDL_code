function ker_mat = calc_kernel(XT_Y,ker_params)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% Compute kernel of input Gram matrix
%
% INPUT:
% XT_Y       - gram matrix of two input matrices X,Y -> (X'*Y)
% ker_params - structure containing kernel parameters
%
% OUTPUT:
% ker_mat    - calculated kernel matrix
% ========================================================================

X = ker_params.X;
Y = ker_params.Y;
ker_type = ker_params.ker_type;
ker_param_1 = ker_params.ker_param_1; 
ker_param_2 = ker_params.ker_param_2; 

switch (ker_type)
    case 'Linear'
        ker_mat = XT_Y;
    case 'Polynomial'
        ker_mat = (XT_Y + ker_param_2).^ker_param_1;
    case 'Gaussian'
        X_sum = sum(X.^2,1)';
        Y_sum = sum(Y.^2,1)';
        ker_mat = X_sum*ones(1,size(Y,2)) + ones(size(X,2),1)*Y_sum' - 2*XT_Y;
        ker_mat = exp(-ker_mat/(2*ker_param_1^2));
end
