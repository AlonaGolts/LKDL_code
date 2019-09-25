function Ds = dictlearn_mb_simple(varargin)
% dictlearn_mb    Learn a dictionary by a minibatch RLS-DLA variant. 
%                 i.e. make a model of the input data X which are available
% through a specified m-file or given as a NxL matrix. 
% The resulting dictionary may be stored in a file, see 'dictfile' option.
% 
%   Use of function:
%   ----------------
%   Ds = dictlearn_mb(varargin)
%     Ds is the dictionary as a struct, Ds.D is the actual dictionary
%     Arguments may be given as pairs: 'option',value, ...
%     (or options may be given in one (or more) struct or cell-array)
% 
%   Options for training data, i.e. how to make the training vectors: 
%   --------------------------------------------------------
%   X        input data, NxL matrix
%            or
%   Xmfile   m-file which return data vectors, i.e. X=feval(Xmfile,Xopt);
%            Options i Xmfile should be processed like in this file
%   Xopt     struct with options for Xmfile, ex: Xopt=struct('L',2000);
%            'L' should be number of training vectors to return
%
%   Options for sparse approximation: 
%   --------------------------------------------------------
%   samet    the sparse approximation method to use. It can be the method
%       argument in sparseapprox.m. Default 'javaormp' 
%       also convex approximation methods: cafixed, convexsets, greedy
%   saopt    the additional options to use in sparseapprox.m as a struct. 
%       Default is: struct('tnz',3, 'verbose',0); 
%   K   the number of atoms in the dictionary, default 100
%
%   Options for minibatch variant of RLS-DLA: 
%   --------------------------------------------------------
%   mb, minibatch   an array with number of batches and number of vectors in each batch.
%       Default: repmat([1000,1],5,1).*[2,10; 2,40; 3,100; 3,200; 2,500];
%       Total number of training vectors is: sum( minibatch(:,1).*minibatch(:,2) )
%   lam0, lambda0   The initial value of the forgetting factor lambda, default 0.996.
%   lam1, lambda1   when lambda should be increased to 1, given as a number relative
%       to the scheduled number of vectors to process, default 0.9.
%   outlim   limit for data outlayers, limit for approximation error relative to
%       mean of the last errors. Default: inf (which is no limit)
%   D    initial dictionary, default: random training vectors
%   A    initial A matrix, default: identity matrix
%
%   Options for mat-files to store dictionary in (or retrieve from): 
%   --------------------------------------------------------
%   dictin, optionfile   name of mat-file where options are stored.
%       This can be a dictionary file.
%   dictfile  file to store Ds in. If not given Ds will not be store
%
%   Options for logging and displaying results: 
%   --------------------------------------------------------
%   ptc, PropertiesToCheck   a cell array with names of dictionary properties to chech
%       at regular intervals during learning given by checkrate. None: {{}}.
%       Default: {{'mtvp','r2avg','traceA','fbA','fbB','betamse','d2min'}}
%       May also use: 'nofr', 'rcondA', 'snr'
%       Note {{ and }} used to define a cell array as a field in a struct!
%       It is good to have 'mtvp' (or 'tvp') as first property (used as x-axis in plots)
%   checkrate   how often properties are reported and dictionary checked, default 1000
%   v, verbose  verbose level given as 0 (only warnings) 1 (normal) 2 (much)
%
%   Examples:
%   ---------
%   X = load('dataXforAR1.mat');
%   opt = struct('K',32, 'samet','mexomp','saopt',struct('tnz',4));
%   Ds = dictlearn_mb('X',X, opt);    
%   figure(1);clf;plot(Ds.ptab(:,1),Ds.ptab(:,2),'b-');xlabel(Ds.ptc{1});ylabel(Ds.ptc{2}); 

%----------------------------------------------------------------------
% Copyright (c) 2013.  Karl Skretting.  All rights reserved.
% University of Stavanger, Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  08.04.2013  Made function (based on texdictlearn.m)
%----------------------------------------------------------------------

mfile = 'DictLearn_MB';

%% defaults, initial values
tstart = tic;
text = [mfile,' started ',datestr(now)];  
dictfile = '';
Xmfile = '';
Xopt = struct('L',2000);
samet = 'javaORMP';
K = 32;
fac2K = 250;   % only used to set minibatch below
minibatch = repmat([fac2K,1],5,1).*[2,10; 2,40; 3,100; 3,200; 2,500];  
lam0 = 0.99;
lam1 = 0.9;
tvp = 0;    % number of training vectors processed
outnum = 0; 
outlim = inf;
D = [];
A = [];
B = [];
ptc = {'mtvp','r2avg','traceA','fbA','fbB','betamse','d2min'};
checkrate = 1000;
verbose = 1;
snr = 0;   % used only when X is given
ptab = 0;

%% get the options
nofOptions = nargin;
optionNumber = 1;
fieldNumber = 1;
while (optionNumber <= nofOptions)
    if isstruct(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        sNames = fieldnames(sOptions);
        opName = sNames{fieldNumber};
        opVal = sOptions.(opName);
        % next option is next field or next (pair of) arguments
        fieldNumber = fieldNumber + 1;  % next field
        if (fieldNumber > numel(sNames)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    elseif iscell(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        opName = sOptions{fieldNumber};
        opVal = sOptions{fieldNumber+1};
        % next option is next pair in cell or next (pair of) arguments
        fieldNumber = fieldNumber + 2;  % next pair in cell
        if (fieldNumber > numel(sOptions)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    else
        opName = varargin{optionNumber};
        opVal = varargin{optionNumber+1};
        optionNumber = optionNumber + 2;  % next pair of options
    end
    % interpret opName and opVal
    if  strcmpi(opName,'X') 
        if isnumeric(opVal)
            Xin = opVal;
        else
            error([mfile,': illegal type (class) of value for option ',opName]);
        end
    end
    if  strcmpi(opName,'Xmfile') 
        if ischar(opVal)
            if exist(opVal, 'file')
                Xmfile = opVal;
            else
                error([mfile,': can not find ',opVal]);
            end
        else
            error([mfile,': illegal type (class) of value for option ',opName]);
        end
    end
    if  strcmpi(opName,'Xopt') 
        if isstruct(opVal)
            Xopt = opVal;
        else
            error([mfile,': illegal type (class) of value for option ',opName]);
        end
    end
    %
    if ( strcmpi(opName,'samet') )
        if ischar(opVal)
            samet = opVal;
        else
            error([mfile,': not character value (string) for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'saopt') )
        if isstruct(opVal)
            saopt = opVal;
        else
            error([mfile,': not struct for option ',opName]);
        end
    end 
    if strcmpi(opName,'K')    
        if isnumeric(opVal)
            K = opVal;
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    %
    if ( strcmpi(opName,'minibatch') || strcmpi(opName,'mb') )
        if isnumeric(opVal)
            minibatch = opVal;
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'lam0') || strcmpi(opName,'lambda0') )
        if isnumeric(opVal)
            lam0 = opVal(1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'lam1') || strcmpi(opName,'lambda1') )
        if isnumeric(opVal)
            lam1 = opVal(1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'r2avg') || strcmpi(opName,'r2average') )
        if isnumeric(opVal)
            r2avg = opVal(1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'outlim') || strcmpi(opName,'outlimit') )
        if isnumeric(opVal)
            outlim = opVal(1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'outnum') || strcmpi(opName,'outnumber') )
        if isnumeric(opVal)
            outnum = opVal(1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if strcmpi(opName,'D') 
        if isnumeric(opVal)
            D = opVal;
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if strcmpi(opName,'A') 
        if isnumeric(opVal)
            A = opVal;
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if strcmpi(opName,'B')
        if isnumeric(opVal)
            B = opVal;
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end
    %
    if strcmpi(opName,'dictin') || strcmpi(opName,'optionfile')
        if ischar(opVal)
            dictin = opVal;
            if ((numel(dictin) < 5) || ~strcmpi(dictin((end-3):end),'.mat'))
                dictin = [dictin,'.mat']; %#ok<AGROW>
            end
            if (~exist(dictin, 'file') &&  (numel(strfind(dictin,cat_sep)) == 0))
                dictin = [cat_D,cat_sep,dictin];   %#ok<AGROW>
            end
            if exist(dictin, 'file')
                disp([mfile,': use all fields in ',dictin,' as options.']);
                disp('  !!!  NOTE this is NOT tested yet, it may work though. ');
                % but some few should be ignored
                mfileHERE = mfile;
                tstartHERE = tstart;
                textHERE = text;
                clear text
                load(dictin);       % overwrite variables in the m-file!
                mfile = mfileHERE;  
                tstart = tstartHERE;
                if exist('text','var')
                    text = char(text, textHERE);
                else
                    text = textHERE;
                end
                dictfile = dictin;
                % the more careful code would be like
                %  Ds = load(dictin);  
                %  if isfield(Ds,'imno'); imno = Ds.imno; end;
                %  if isfield(Ds,'imfile'); imfile = Ds.imfile; end;
                %  ....
            else
                error([mfile,': can not find ',dictin]);
            end
        else
            error([mfile,': not char for option ',opName]);
        end
    end
    if strcmpi(opName,'dictfile') || strcmpi(opName,'dictout')
        if ischar(opVal)
            dictfile = opVal;
        else
            error([mfile,': not char for option ',opName]);
        end
    end
    %
    if strcmpi(opName,'tvp') 
        if isnumeric(opVal)
            tvp = opVal(1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if strcmpi(opName,'text') 
        if ischar(opVal)
            text = char(opVal, text);
        else
            error([mfile,': not char for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'PropertiesToCheck') || strcmpi(opName,'ptc') )
        if iscell(opVal)
            ptc = opVal;
        else
            error([mfile,': not cell for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'PropertiesTable') || strcmpi(opName,'ptab') )
        if isnumeric(opVal)
            ptab = opVal;
            pti = size(ptab,1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if ( strcmpi(opName,'checkrate') || strcmpi(opName,'cr') )
        if isnumeric(opVal)
            checkrate = opVal(1);
        else
            error([mfile,': not numeric for option ',opName]);
        end
    end 
    if strcmpi(opName,'verbose') || strcmpi(opName,'v')
        if (islogical(opVal) && opVal); verbose = 1; end;
        if isnumeric(opVal); verbose = opVal(1); end;
    end
end

%% some important variables are checked
K = K(1);
if (~exist('saopt','var') || ~isstruct(saopt) || ~isfield(saopt,'tnz'))
    saopt = struct('tnz',3, 'verbose',0);
end

%%
if exist('Xin','var')   % 
    [N,L] = size(Xin);
else
    if ~(exist(Xmfile,'file') == 2)
        error([mfile,': training data not available.']);
    end
    if ~isfield(Xopt,'L')
        Xopt.L = checkrate;
    end    
    L = Xopt.L;
end

%% initialize variables, including dictionary
if (size(D,1) == N) && (size(D,2) == K) 
    % D is probably ok
else
    if exist('Xin','var')   % 
        D = Xin(:,(L-K+1):L);   % the last ones
    else
        D = feval(Xmfile,Xopt,'L',K);
        if (size(D,2) > K)     % make sure D does not have to many atoms
            D = D(:,1:K);
        end
    end
end

% normalization of dictionary
g = 1./sqrt(sum( D.*D ));
D = D .* repmat(g, N, 1); % normalize dictionary

if (numel(A) == 1)
    B = D/A;   % note A is scalar here
    A = eye(K)*A;
end
if ~((size(A,1) == K) && (size(A,2) == K)) 
    A = eye(K);
    B = D;
end
tvtot = tvp + sum( minibatch(:,1).*minibatch(:,2) );

r2avg = 0;


%%  minibatch variant of RLS-DLA
if exist('Xin','var')   % 
    X = Xin;
else
    X = feval(Xmfile,Xopt);
end
[N,L] = size(X);
xno = 0;
for linje = 1:size(minibatch,1)
%     disp([num2str(linje), ' out of ',num2str(size(minibatch,1))]);
    for bno = 1:minibatch(linje,1)
%         disp([num2str(bno),' out of ',num2str(minibatch(linje,1))]);
        batchsize = minibatch(linje,2);
        if exist('Xin','var')   % 
            if rand(1) < 0.25
                idx = rem(xno+(1:batchsize)-1,L)+1;
                xno = idx(end);
            else
                idx = ceil(rand(1,batchsize)*L);   % just random vectors
            end
            Xbatch = X(:,idx);
        else
            if (xno+batchsize) > size(X,2)        % get more training vectors
                X = feval(Xmfile,Xopt);
                xno = 0;
            end
            Xbatch = X(:,xno+(1:batchsize));
            xno = xno+batchsize;
        end
        
        W = omp(D'*Xbatch, D'*D, saopt.tnz);
        
%         W = sparseapprox(Xbatch, D, samet, saopt);

%         R = Xbatch - D*W;
        lam = lambdafun(tvp, 'Q', lam1*tvtot, lam0, 1).^batchsize;
%         u = lam*A*W;
%         v = D'*R;
%         alpha = 1./(1+(W')*u);
%         D = D + alpha*R*u';
%         A = lam*A - alpha*(u*u');
        A = lam*A + full(W*W');
        B = lam*B + full(Xbatch*W');
        D = B/A;
        g = 1./sqrt(sum( D.*D ));
        D = D .* repmat(g, N, 1); % normalize dictionary, not A and B though
        tvp = tvp+batchsize;
    end
end

%% set output argument
tu = toc(tstart);
tuh = floor(tu/3600);
tum = floor(tu/60) - tuh*60;
tus = ceil(tu) - tum*60 - tuh*3600;
t1 = sprintf('%s finished %s, time used is %i:%02i:%02i (hh:mm:ss), %6.3f ms/tvp.', ...
         mfile, datestr(now), tuh, tum, tus, tu*1000/tvp );
text = char(text, t1);
Ds = struct( 'dictfile', dictfile, 'mfile', mfile, ...
             'Xmfile', Xmfile, 'Xopt', Xopt, ...
             'samet', samet, 'saopt', saopt, ...
             'lam0', lam0, 'lam1', lam1, ...
             'N', N, 'K', K, 'L',L, 'D', D, 'A', A, 'B', B,...
             'r2avg', r2avg, ...
             'ptc', {ptc}, 'ptab', ptab, 'checkrate', checkrate, ...
             'minibatch', minibatch, 'tvp', tvp, ...
             'timeused', tu, ...
             'text', text );