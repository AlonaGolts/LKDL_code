classdef KSVDCoresetAlg < handle
    % KSVDCORESET Create a Coreset for computing the KSVD algorithm
	% This is an implementation of the Coreset described in:
	%
	% Feldman D., Feigin M. and Sochen N., "Learning Big (Image) Data via Coresets for Dictionaries", in Journal of Mathematical Imaging and Vision, March 2013
	% Feigin M., Feldman D. and Sochen N, "From High Definition Image to Low Space Optimization", in Scale Space and Variational Methods (SSVM), Ein-Gedi, Israel, 2011
    %
	% In order to use the code one needs to
	% 1. Construct a class of type KSVDCoresetAlg
	% 2. Optionally change the default properties
	% 3. Use the computeCoreset function to compute a Coreset on which to learn the dictionary
	% 4. Run KSVD to learn the Coreset on the dictionary
	% 5. Use OMP or another method to compute the coefficients for the dictionary with respect to the original (full) data
	%
    % NOTE: the vectors used for the approximation need to be inserted into
    % the dictionary for this approximation to be correct
	%
	% See computeDictionary function for an example

    properties
        sampleSize = 300; % Number of samples in the Coreset
        svdVecs = 0; % number of singular vectors to use for the optimal subspace approximation
                     % Set to zero to use the constant vector
    end
    
    methods
        % Constructor
        function obj = KSVDCoresetAlg()
        end

        % Compute a Coreset for computing the dictionary for a set of vectors
		% Input:
		% P - (n X d) input dataset, n elements of dimension d are in rows
		%
		% Output:
		% Coreset - the Coreset matrix over which to compute the dictionary using KSVD, elements are in rows
		%           Note: the KSVD algorithm provided by Ron Rubinstein requires the transpose of this
		% D - The base dictionary approximation, this should be added to the final computed dictionary
		% weights - the weights vectors applied to the elements in Coreset
		%
		% Steps in this function :
		%
		% 1. Compute an approximation to the ideal dictionary
		%    Either a constant vector or a set of the largest svdVecs vectors
		%    of the SVD
		% 2. Compute the distances of each of the input vectors from the
		%    dictionary dists = sum((P - P * D * D').^2, 2)
		% 3. Randomly sample a Coreset from the residual (P - P * D * D') with
		%    probability proportional to dists
		% 4. Set weights = 1 / (|C| * probability)
		% 5. Multiply each vector in the Coreset by the square root of it's
		%    weight (correct for L2 distance)

        function [coreset, D, weights,indexes] = computeCoreset(obj, P)
            d = size(P, 2);

            if (obj.svdVecs >= 1)
				% Note: the use of SVDCoresetAlg to compute the SVD can be replaced here with a full SVD
				% For that, replace the following 4 lines
				% [~, ~, D] = svds(P, obj.svdVecs);

                k = SVDCoresetAlg();
                k.beta = obj.svdVecs;
                C = k.computeCoreset(P);

                [~, ~, D] = svds(C, obj.svdVecs);
                
                % TODO: this can be simplified a bit for the case of
                % svdVecs == 1
                R = P - P*D*D';

                clear k C;
            else
                D = ones(size(P, 2), 1)/sqrt(d);
                R = bsxfun(@minus, P, mean(P, 2));
            end
            
            dists = sum(R .* R, 2);
            dists(dists < 0) = 0;

            sumDists = sum(dists);
            probs = dists ./ sumDists;
            
            % Perform random selection with returns of obj.sampleSize with
            % the given probability distribution
            % This seems to be a bit faster than using randsample

            bins = histc(rand(obj.sampleSize, 1), [0 ; cumsum(probs)]);
            indexes = bins(1:end-1) ~= 0;

            % Under the L2 norm we can multiply the vector by the sqrt of the weight

            % temporary change (Alona)
            weights = 1 / obj.sampleSize * bins(indexes) ./ probs(indexes);
            sqrtWeights = sqrt(weights);
            
%             coreset = bsxfun(@times, sqrtWeights, R(indexes, :));
            coreset = P(indexes,:);
        end
        
        % Compute the dictionary and coefficients for an input data set
		% Input:
		% P - (n x d) matrix, with vectors of dimension d in rows
        % T - The sparsity measure (how many vectors to use for
        %     reconstruction)
		% dictsize - The requested dictionary size,
        % iternum - The number of KSVD iterations to run on the corset
        %
		% Output:
		% D - The output dictionary, with vectors in columns
		% X - the coeffiecient matrix, that is the reconstruction is P = DX
		%
		% Note: This function requires the KSVD code from Ron Rubinstein 
		%       http://www.cs.technion.ac.il/~ronrubin/software.html
        function [D, X] = computeDictionary(obj, P, T, dictsize, iternum)
            if ~exist('T', 'var') || isempty(T)
                T = 5;
            end
            if ~exist('dictsize', 'var') || isempty(dictsize)
                dictsize = 100;
            end
            if ~exist('iternum', 'var') || isempty(iternum)
                iternum = 20;
            end

            params.Tdata = T;
            params.iternum = iternum;
            
            if obj.svdVecs
                params.dictsize = dictsize - obj.svdVecs;
            else
                params.dictsize = dictsize - 1;
            end
            
            [C tD] = obj.computeCoreset(P);
            
            params.data = C';
            
            [D, ~] = ksvd(params, '');
            D = [D tD];
            X = omp(D, refData, D'*D, params.Tdata);
        end
    end

    methods(Static)
        % Function to run some measurement tests
        function Test(P)
            rng('default')

            % Test parameters
            T = 5;
            dictsize = 100;
            iternum = 20;
            sampleSize = 200:500:10000;
            svdVecs = 0:3;

            % Output parameters
            filename = 'results.xls';
            startrow = 1;
            sheet = 1;

            refData = P';

            params.Tdata = T;
            params.dictsize = dictsize;
            params.iternum = iternum;
            params.data = refData;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Run KSVD on original data
            %%%%%%%%%%%%%%%%%%%%%%%%%%%

            t = tic;
            [D, X] = ksvd(params, '');
            t = toc(t);
            err = sum(sum((refData - D*X).^2)) / numel(P);

            xlswrite(filename, {'Datasize', 'Data dimension', 'dictionary size', 'sparsity', 'error', 'time'}, sheet, sprintf('A%d', startrow));
            xlswrite(filename, {size(P, 1), size(P, 2), dictsize, T, err, t}, sheet, sprintf('A%d', startrow + 1));
            startrow = startrow + 3;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Run KSVD on Coreset
            %%%%%%%%%%%%%%%%%%%%%%%%%%%

            k = KSVDCoresetAlg();
            
            for nvecs = svdVecs
                fprintf('starting nvecs = %g\n', nvecs);

                k.svdVecs = nvecs;
                if nvecs
                    params.dictSize = dictsize - nvecs;
                else % A single constant vector
                    params.dictSize = dictsize - 1;
                end
                
                i = 1;
                t1 = zeros(length(sampleSize), 1);
                t2 = zeros(length(sampleSize), 1);
                t3 = zeros(length(sampleSize), 1);
                t4 = zeros(length(sampleSize), 1);
                err = zeros(length(sampleSize), 1);
                err1 = zeros(length(sampleSize), 1);
                
                for s = sampleSize
                    fprintf('starting samplesize = %g\n', s);

                    % Run on Coreset of size sampleSize

                    k.sampleSize = s;
                    t = tic;
                    [C tD] = k.computeCoreset(P);
                    t1(i) = toc(t);
                    
                    params.data = C';
                    
                    t = tic;
                    [D, ~] = ksvd(params, '');
                    D = [D tD];
                    X = omp(D, refData, D'*D, params.Tdata);
                    t2(i) = toc(t);
                    
                    err(i) = sum(sum((refData - D*X).^2)) / numel(P);
                    
                    % run on a random sample of the same size to compare
                    % error and time
                    
                    t = tic;
                    C = P(randsample(size(P, 1), s), :);
                    t3(i) = toc(t);
                    
                    params.data = C';
                    
                    t = tic;
                    [D1, ~] = ksvd(params, '');
                    X1 = omp(D1, refData, D1'*D1, params.Tdata);
                    t4(i) = toc(t);
                    
                    err1(i) = sum(sum((refData - D1*X1).^2)) / numel(P);
                    
                    i = i + 1;
                end
                
                i = i - 1;
                xlswrite(filename, {'Coreset Size', 'svdVecs', 'Coreset error', 'Coreset time', 'KSVD time', 'total time', 'Random sample error', 'Random sample time', 'Random sample KSVD time', 'Random sample total time'}, sheet, sprintf('A%d', startrow));
                xlswrite(filename, sampleSize(:), sheet, sprintf('A%d:A%d', startrow + 1, startrow + i));
                xlswrite(filename, nvecs, sheet, sprintf('B%d:B%d', startrow + 1, startrow + i));
                xlswrite(filename, err, sheet, sprintf('C%d:C%d', startrow + 1, startrow + i));
                xlswrite(filename, t1, sheet, sprintf('D%d:D%d', startrow + 1, startrow + i));
                xlswrite(filename, t2, sheet, sprintf('E%d:E%d', startrow + 1, startrow + i));
                xlswrite(filename, t1 + t2, sheet, sprintf('F%d:F%d', startrow + 1, startrow + i));
                xlswrite(filename, err1, sheet, sprintf('G%d:G%d', startrow + 1, startrow + i));
                xlswrite(filename, t3, sheet, sprintf('H%d:H%d', startrow + 1, startrow + i));
                xlswrite(filename, t4, sheet, sprintf('I%d:I%d', startrow + 1, startrow + i));
                xlswrite(filename, t3 + t4, sheet, sprintf('J%d:J%d', startrow + 1, startrow + i));
                
                startrow = startrow + i + 2;
            end
        end
    end
end