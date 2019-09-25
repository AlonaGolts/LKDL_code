function [opts] = Comp_CentSparseCoding3_largedata (ipts,par)
%   
%    This function is to sovle the sparse coding problem 
%    using Iterative projection Method
%
%      Inputs
%             ipts        -    the input data structure
%               .D         -    dictoanry
%               .X         -    the traing data matrix,
%                               each column is a sample
%             par         -    the input para structure
%               .tau       -    the l1 constraint's para
%               .lambda    -    the l2 minus mean's para
%               .nIter     -    the maximal iteration num
%               .sigma     -    the parameter
%               .isshow    -    whether plot the gap
%               .initM     -         initial method
%               .twist     -         true: twist acceleration;false:no
%                                          accleration
%               .citeT    -         the relative citeration threhold,
%                                          default:1e-4
%               .cT        -         the stop citeration of c, the
%                                          threhold
%   
%      Outputs
%               A         -    the sparse coefficient
%
%   the model is
%     argmin{
%            ||X-DA||_F^2+ 2*tau1||A||_1+lambda*||AB||_F^2
%            }
%            B is defined below.
%     
%    written by Mike Yang.
%

m    =    size(ipts.D,2);
n    =    size(ipts.X,2);

switch lower(par.initM)
    case {'zero'}
        A    =    zeros(m,n);
    case {'transpose'}
        A  =  ipts.D'*ipts.X;
    case {'pinv'}
        A  =  pinv(ipts.D)*ipts.X;
    case {'last'}
        A    =    ipts.last_coef;
    otherwise
        error('Nonknown method!');
end

D        =    ipts.D;
X        =    ipts.X;
tau      =    par.tau;
lambda   =    par.lambda;
nIter    =    par.nIter;
% sigma    =    par.sigma*find_max_eigenv(D'*D);
c        =    par.c;
sigma    =    c;
twist    =    par.twist;
tau1     =    tau/2;
% B        =    eye(n)-ones(n,n)/n;

% At_pref   =    A(:);
% At_now    =    A(:);
At_pref   =    A;
At_now    =    A;

diag_D    =    [];
diag_B    =    [];

for_ever           =         1;
IST_iters          =         0;
TwIST_iters        =         0;
sparse             =         1;
verbose            =         1;
enforceMonotone    =         1;
lam1               =         1e-4;   %default eigenvalues
lamN               =         1;      %default eigenvalues
rho0               =         (1-lam1/lamN)/(1+lam1/lamN); 
alpha              =         2/(1+sqrt(1-rho0^2));        %default,user can set
beta               =         alpha*2/(lam1+lamN);         %default,user can set
%-------------------------------------------------------------
%start of main loop
%-------------------------------------------------------------
if ~twist
for nit  =  2: nIter

    v1   =    [];
    for i  =   1:n
        A     =   reshape(At_pref,[m,n]);
        tem1  =   X(:,i)-D*A(:,i);
        tem2  =   D'*tem1;
        v1    =   [v1;tem2];
    end
    A     =   reshape(At_pref,[m,n])';
    v2_2   =   [];
    for i  =   1:m
         tem1  =  B*A(:,i);
         v2_2  =  [v2_2;tem1];
    end
    v2_3  =  reshape(v2_2,[n m])';
    v2    =  v2_3(:);
%     v1    =  diag_D'*(X(:)-diag_D*At(:,nit-1));
%     v2_1  =  reshape(At(:,nit-1),[m,n])';
%     v2_2  =  diag_B*v2_1(:);

    v     =  At_pref+(v1-lambda*v2)/sigma;
    At_now  =  soft(v,tau1/sigma);
    At_pref  =  At_now;
    
    A     =   reshape(At_now,[m,n]);
    gap1  =   norm((X-D*A),'fro')^2;
    if n==1
        gap2 = norm(A*B,2)^2;
    else
        gap2  =   norm(A*B,'fro')^2;
    end
    
    gap3  =   sum(abs(A(:)));
    ert(nit-1)   =   gap1+2*tau1*gap3+lambda*gap2;
    fprintf(['Iteration num:' num2str(nit-1) '    Gap:' num2str(ert(nit-1)) '\n']);
end

else
   xm2       =      At_pref;
   xm1       =      At_pref;
   
%    A     =   reshape(At_pref,[m,n]);
   A     =   At_pref;
   gap1  =   norm((X-D*A),'fro')^2;
   if n==1
       meanA = mean(A,2); 
       gap2 = norm(A-meanA,2)^2;
   else
       meanA = mean(A,2);
       gap2  =   norm(A-meanA*ones(1,n),'fro')^2;
   end
   gap3  =   sum(abs(A(:)));
   prev_f   =   gap1+2*tau1*gap3+lambda*gap2;
   
   for n_it = 2 : nIter;

%           A     =   reshape(At_now,[m,n]);
          A     =   At_now;
          gap1  =   norm((X-D*A),'fro')^2;
          if n==1
            meanA = mean(A,2); 
            gap2 = norm(A-meanA,2)^2;
          else
            meanA = mean(A,2);
            gap2  =   norm(A-meanA*ones(1,n),'fro')^2;
          end
          gap3  =   sum(abs(A(:)));
          ert(n_it-1)   =   gap1+2*tau1*gap3+lambda*gap2;
  
%    fprintf('Iteration:%f  Total gap:%f\n',n_it,ert(n_it-1));
    
    while for_ever
        % IST estimate
%     v1   =    [];
%     for i  =   1:n
%         A     =   reshape(xm1,[m,n]);
%         tem1  =   X(:,i)-D*A(:,i);
%         tem2  =   D'*tem1;
%         v1    =   [v1;tem2];
%     end
%     A  = reshape(xm1,[m,n]);
    A  = xm1;
    v1 = D'*(X-D*A); 
%     v1 = v1(:);
    
%     A     =   reshape(xm1,[m,n])';
    A     =   xm1';
%     v2_2   =   [];
%     for i  =   1:m
%          tem1  =  B*A(:,i);
%          v2_2  =  [v2_2;tem1];
%     end
%     v2_3  =  reshape(v2_2,[n m])';
    v2_2  =  A - ones(n,1)*mean(A,1); 
%     v2_3 = v2_2';
%     v2    =  v2_3(:);
    v2 = v2_2';
    
    
        v     =  xm1+(v1-lambda*v2)/sigma;
        x_temp  =  soft(v,tau1/sigma);
        
        if (IST_iters >= 2) | ( TwIST_iters ~= 0)
            % set to zero the past when the present is zero
            % suitable for sparse inducing priors
            if sparse
                mask    =   (x_temp ~= 0);
                xm1     =   xm1.* mask;
                xm2     =   xm2.* mask;
            end
            % two-step iteration
            xm2    =   (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x_temp;
            % compute residual
            
%             A     =   reshape(xm2,[m,n]);
            A     =   xm2;
            gap1  =   norm((X-D*A),'fro')^2;
            if n==1
             meanA = mean(A,2); 
             gap2 = norm(A-meanA,2)^2;
            else
             meanA = mean(A,2);
             gap2  =   norm(A-meanA*ones(1,n),'fro')^2;
            end
            gap3  =   sum(abs(A(:)));
            f   =   gap1+2*tau1*gap3+lambda*gap2;
          
            if (f > prev_f) & (enforceMonotone)
                TwIST_iters   =  0;  % do a IST iteration if monotonocity fails
            else
                TwIST_iters =   TwIST_iters+1; % TwIST iterations
                IST_iters   =    0;
                x_temp      =   xm2;
                if mod(TwIST_iters,10000) ==0
                   c = 0.9*c; 
                   sigma= c;
                end
                break;  % break loop while
            end
        else
%           A     =   reshape(x_temp,[m,n]);
          A     =   x_temp;
          gap1  =   norm((X-D*A),'fro')^2;
          if n==1
            meanA = mean(A,2); 
            gap2 = norm(A-meanA,2)^2;
          else
            meanA = mean(A,2);
            gap2  =   norm(A-meanA*ones(1,n),'fro')^2;
          end
          gap3  =   sum(abs(A(:)));
          f   =   gap1+2*tau1*gap3+lambda*gap2;

            if f > prev_f
                % if monotonicity  fails here  is  because
                % max eig (A'A) > 1. Thus, we increase our guess
                % of max_svs
                c         =    2*c; 
                sigma     =    c;
                if verbose
                    fprintf('Incrementing c=%2.2e\n',c);
                end
                if  c > par.cT
                    break;  % break loop while    
                end
                IST_iters = 0;
                TwIST_iters = 0;
            else
                TwIST_iters = TwIST_iters + 1;
                break;  % break loop while
            end
        end
    end

    citerion      =   abs(f-prev_f)/prev_f;
    if citerion < par.citeT | c > par.cT
%        fprintf('Stop!\n c=%2.2e\n citerion=%2.2e\n',c,citerion);
       break;
    end
    
    xm2           =   xm1;
    xm1           =   x_temp;
    At_pref      =   At_now;
    At_now       =   x_temp;
    prev_f        =   f;
    
   end

end
%--------------------------------------------------------------------------
% end of main loop
%--------------------------------------------------------------------------
% opts.A     =       reshape(At_now,[m,n]);
opts.A     =       At_now;
opts.ert   =       ert;
if par.isshow
    plot(ert,'r-');
end