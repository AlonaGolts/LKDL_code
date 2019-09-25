function [D,mCoef,er]   =   CDLl2_DictLearn_general(X,D_init,tau,lambda,nIter,isShow)
%
%   l1 Sparse Dictionary Learn Function
%   using function double_l1.m
%  
%               Inputs:  
%                     X      -        traindata matrix. Each column vector
%                                     of X is a sample
%                     D_init -        the initilaized dictionary
%                     .tau       -    the l1 constraint's para
%                     .lambda    -    the l2 minus mean's para
%                     nIter  -        Iteration numbers
%                     isShow -        whether show error
%               Outputs:
%                     D      -        learnt dictionary
%                     mCoef  -        mean vector of coef
%                     er     -        vector of gap
%
%   Written by Mike Yang (csmyang AT comp.polyu.edu.hk)


n_it            =     0;
mCoef           =     [];
D               =     D_init;
er_now          =     0;
er              =     [];

par.initM      =     'zero';  % 'zero','transpose','pinv',in double_l1
ipts.last_coef = zeros(size(D,2),size(X,2));
par.nIter       =     200;     %in function double_l1
par.sigma       =     1.05;     %in function double_l1
par.isshow      =     false;
par.twist      =     true;
par.citeT      =     1e-6;
par.cT         =     1e+10;

while n_it < nIter
%     fprintf(['Iteration=' num2str(n_it) '\n']);
    TemD=D;
if size(TemD,1)>size(TemD,2)
      par.c        =    par.sigma*find_max_eigenv(TemD'*TemD);
else
      par.c        =    par.sigma*find_max_eigenv(TemD*TemD');
end
    ipts.D      =   D;
    ipts.X      =   X;
    par.tau     =   tau;
    par.lambda  =   lambda;
%     tic
%     [opts] = Comp_CentSparseCoding3 (ipts,par);
%     toc
    [opts] = Comp_CentSparseCoding3_largedata (ipts,par);
    coef   = opts.A;  
 
    % Fix alpha, update D.
    newD        =   [];
    newcoef     =   [];  
    for i =  1:size(D,2)
    ai      =    coef(i,:);
    Y       =    X-D*coef+D(:,i)*ai;
    di      =    Y*ai';
    if norm(di,2) < 1e-6
       di        =    zeros(size(di));
    else
       di        =    di./norm(di,2);
       newD      =    [newD di];
       newcoef   =    [newcoef;ai];
    end
    D(:,i)  =    di;
    end
    
    D       =    newD;
    coef    =    newcoef;
    ipts.last_coef = coef;
    
    mCoef        =   mean(coef')';
    mean_alpha   =   repmat(mean(coef')',[1 size(coef,2)]);
    zz           =   X-D*coef;
    l22          =   0;
    
    for tem_j    =   1:size(coef,1)
        l22      =   l22+norm(coef(tem_j,:)-mean_alpha(tem_j,:),2)^2;
    end
    
    er_now       =   zz(:)'*zz(:)+tau*sum(abs(coef(:)))+lambda*l22;
    n_it         =   n_it +1;
    er           =   [er er_now];
    
end

if isShow
    plot(er,'r-*');
end