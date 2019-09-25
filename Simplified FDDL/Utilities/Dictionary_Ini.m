function D    =    Dictionary_Ini(data,nCol,wayInit)
%     
%   Dictionary initialization function
%
%        Inputs
%          data         - data matrix. Each column vector of data
%                        a sample vector of the same class
%          nCol         - the number of Dict's columns.
%          wayInit      - the initial method of dictionary.
%        Outputs
%          D            - initilazed dictionary
%
%    writeen by Mike Yang (csmyang AT comp.polyu.edu.hk)

m   =    size(data,1);

switch lower(wayInit)
    case {'pca'}
        [D,disc_value,Mean_Image]   =    Eigenface_f(data,nCol-1);
        D                           =    [D Mean_Image./norm(Mean_Image)];
    case {'random'}
        phi                         =    randn(m, nCol);
        phinorm                     =    sqrt(sum(phi.*phi, 1));
        D                           =    phi ./ repmat(phinorm, size(phi,1), 1);
    case {'kmeans'}
        [idx,C] = kmeans(data',nCol);
        D       = C';
        D  =  D./( repmat(sqrt(sum(D.*D)), [size(D,1),1]) );
    case {'partial'}
        idx = randperm(nCol);
        D = data(:,idx(1:nCol));
        D  =  D./( repmat(sqrt(sum(D.*D)), [size(D,1),1]) );
    otherwise 
        error{'Unkonw method.'}
end
        
