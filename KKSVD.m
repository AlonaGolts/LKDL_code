function [A, X] = KKSVD(params)
% Input : 
%   	  data = column data; dictsize = number of atoms
%   	  Tdata = sparsity level; iternum = number of iterations
% 		  kernel = {'linear','poly','gauss','hint'} ; 
%		  kervar1, kervar2 = kernel parameters (see 'gram.m' for details)
% Goal  : Learn D and X by min || Phi(Y) - D*X ||_2 , st ||x_i|| <= T0
% Output: Dictionary coefficients A, where D = Phi(Y)*A 
% 	      Sparse coefficients X

Y = params.data; kernel_choice = params.kernel; 
T0 = params.Tdata; iternum = params.iternum ; 
ds = params.dictsize; 
% init_dic = params.initdic;
kervar1 = params.kervar1; kervar2 = params.kervar2; 

K = gram(Y',Y',kernel_choice, kervar1, kervar2); % compute Gram matrix

mynorm = Knorms(eye(size(K,1)),K) ; 
mynorm = mynorm(:) ;
mynorm = (mynorm*mynorm') ; 
K = K./mynorm ; % normalize to norm-1 in feature space


samplenum = size(Y,2) ; 
D = zeros(samplenum, ds) ; 
randid = randperm(samplenum); 

% if isempty(init_dic)
    for i=1:ds
        D(randid(i),i) = 1;  % randomly initilize dictionary
    end
% else
%     D = init_dic;
% end

total_err = zeros(iternum,1) ;
for it=1:iternum

%%%% do not print the progress of DL (Alona)
% fprintf('%d...',it);  if (mod(it,30)==0), fprintf('\n'); end

[X] = KOMP(D, K, K, T0) ; 

A = eye(samplenum) - D*X ; 
proj_err = Knorms(A,K) ; % norm of columns in feature space

[val LeastRepId] = sort(proj_err, 'descend') ; 

total_err(it) = sum(proj_err) ; % projection error

auxD = zeros(size(D)) ; 
for c = 1:ds   
   nonzero = abs(X(c,:) ) >  0 ;  % index of non-zero
      
   if (isempty(find(nonzero, 1)))         
      % replace atom 'dc' with least represented signals  
      % equivalently replace row 'Least' and column 'c' of auxD
      auxD(:,c) = zeros(samplenum,1) ; 
      auxD(LeastRepId(1),c) =  ...
          1/sqrt(K(LeastRepId(1),LeastRepId(1)));  % K(least,least) normalizes columns
      LeastRepId(1) = [] ; 
      continue; 
   end   
   
   redE = A(:,nonzero) + D(:,c)*X(c,nonzero) ; % compute redE by shrinking matrix 
   Gram_redE = redE'*K*redE ;       
      
   % perform SVD 
   [U S V] = svd(Gram_redE) ;         
   
   % keep the first eigenvector of U as dictionary atom
   cand =  redE*U(:,1)/sqrt(S(1,1)) ;    
   
   % update dictionary column and prune similar atoms
   simi = abs(cand'*K*auxD(:,1:c)) ; 
   if (find(simi>0.98, 1))
        auxD(:,c) = zeros(samplenum,1) ; 
        auxD(LeastRepId(1),c) =  ...
          1/sqrt(K(LeastRepId(1),LeastRepId(1))); 
        LeastRepId(1) = [] ;
   else
        auxD(:,c) = cand ; 
   end      
end

   D =auxD ; 

end

%%%% Do not print the progress of DL (Alona)
% fprintf('\n');

X = KOMP(D, K, K, T0) ;

A = D ; 

end