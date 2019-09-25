function reco_ratio = FDDL_aux(tr_dat,trls,tt_dat,ttls,params)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% This function serves as a wrapper function that performs LKDL
% pre-processing if needed and calls FDDL.
%
% INPUT:
% tr_dat - training set images.
% trls - training set labels.
% tt_dat - test set images.
% ttls - test set labels.
% params - FDDL and LKDL params.
%
% OUTPUT:
% reco_ratio - classification accuracy.
% ========================================================================

%loading path
addpath([cd '/Utilities']);
tem_fd = cd;
par.d_fd          =   'data/';

linear_approx   = params.linear_approx;  % 1 - if LKDL pre-processing is chosen
ker_type        = params.ker_type;       % 'Gaussian'/'Polynomial'/'Linear'
ker_param_1     = params.ker_param_1;    % Kernel hyperparameter
ker_param_2     = params.ker_param_2;    % Second kernel hyperparameter
c               = params.c;              % Percent of samples for Nystrom
k               = params.k;              % Rank of eigen-decomposition of W
samp_method     = params.samp_method;    % 'uniform'/'kmeans'/'diag'/'col_norm'/'coreset'
dic_ini         = params.dic_ini;        % Dictionary initialization method
atoms_per_dic   = params.atoms_per_dic;  % Number of atoms for each sub-class dicionary
 

%seting parameter
par.nClass        =   10;
par.nDim          =   256;
lambda            =   0.1;        %test lambda for sparsity
tau               =   0.005;      % test tau for within-class scatter
par.lambda1Array  =   0.1;
par.lambda2Array  =   0.001;      %train lambda2 for within-class scatter
par.nCOL          =   atoms_per_dic;

par.ID            =   [];
par.nameDatabase  =   'USPS';
par.Dict          =   [];
par.Dlabels       =   [];
par.mCoef         =   [];

%
if linear_approx
    ker_params = struct('X',0,'Y',0,'ker_type',ker_type,'ker_param_1',ker_param_1,'ker_param_2',0);
    [tr_dat, tt_dat] = calc_virtual_map(tr_dat,tt_dat,ker_params,samp_method,c,k);
end


%---------------------Simplified FDDL---------------------------------%
wayInit = dic_ini;
lmd1    = par.lambda1Array;
lmd2    = par.lambda2Array;
nIter   = 5;
isShow  = true;
[WDict,WDlabels,WmCoef] = Simplified_FDDL(tr_dat,trls,par.nCOL,par.nClass,lmd1,lmd2,wayInit,nIter,isShow);
par.Dict = WDict;
par.Dlabels = WDlabels;
par.mCoef = WmCoef;

% ------------------------------classification---------------------------%
% ------------pre-compute for Modified DALM  ----------------------------%
for indClass  =  1:par.nClass
    eye_M     =      eye(sum(par.Dlabels==indClass));
    A         =      [par.Dict(:,par.Dlabels==indClass);sqrt(tau)*eye_M];
    [m,n]     =      size(A);
    norm_b    =      mean(sum(abs(A)));
    beta      =      norm_b/m;
    G{indClass}    = single(A * A' + sparse(eye(m)) * lambda / beta);
    invG{indClass} = inv(G{indClass});
end
        
ID    =  [];
idf       =  [];
for indTest = 1:size(tt_dat,2)
    fprintf(['Totalnum:' num2str(size(tt_dat,2)) 'Nowprocess:' num2str(indTest) '\n']);
    for indClass  =  1:par.nClass
        eye_M     =      eye(sum(par.Dlabels==indClass));
        D         =      [par.Dict(:,par.Dlabels==indClass);sqrt(tau)*eye_M];
        y         =      [tt_dat(:,indTest);sqrt(tau)*par.mCoef((indClass==par.Dlabels))];
        [s, nIter] = SolveDALM_M(G{indClass},invG{indClass}, D, y, 'lambda',lambda,'tolerance',1e-3);
        zz        =  y - D *s;
        gap(indClass)     =  zz(:)'*zz(:);
        gCoef3(indClass)  =  sum(abs(s));
    end
    MixG = lambda*gCoef3+gap;
    index              =  find(MixG==min(MixG));
    par.ID(indTest)    =  index(1);
end
reco_ratio       =  (sum(par.ID==ttls))/length(ttls); 
% disp(['The recognition rate is ' num2str(reco_ratio)]);  
    
