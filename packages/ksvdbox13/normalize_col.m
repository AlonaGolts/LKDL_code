function [normcol] = normalize_col(Y)

normcol = sqrt(sum(Y.^2 , 1)) ; 
normcol = repmat(normcol,size(Y,1),1) ; 
normcol = Y./normcol;  

end