function [WDict,WDlabels,WmCoef] = Simplified_FDDL(tr_dat,trls,nCOL,nClass,lmd1,lmd2,wayInit,nIter,isShow)

% Simplified Fisher Discriminative Dictionary Learning (FDDL), Version 1.0
% Copyright(c) 2014  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for learning the
% Simplified Fisher Discriminative Dictionary from a labeled training data
%
% Please refer to the following paper
%
% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang,"Fisher Discrimination 
% Dictionary Learning for Sparse Representation", In IEEE Int. Conf. on
% Computer Vision, 2011.
% 
% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang, ¡°Sparse representation based Fisher discrimination dictionary learning for image classification¡±, 
% International Journal of Computer Vision (IJCV), 2014.
%----------------------------------------------------------------------
%
%Input : (1) tr_dat:   the training data matrix. 
%                      Each column is a training sample
%        (2) trls:     the training data labels
%        (3) the parameters:
%               nClass   the number of classes
%               nCOL     the number of dictionary atoms
%               wayInit  the way to initialize the dictionary
%               lmd1  the parameter of l1-norm energy of coefficient
%               lmd2  the parameter of l2-norm of Fisher Discriminative
%               coefficient term
%               nIter    the number of FDDL's iteration
%               show     sign value of showing the gap sequence
%
%Output: (1) WDict:  the learnt dictionary via FDDL
%        (2) WDlabels:  the labels of learnt dictionary's columns
%        (2) WmCoef: Mean Coefficient Matrix. Each column is a mean coef
%                   vector

%-----------------------------------------------------------------------

WDict     =   [];
WDlabels  =   [];
WmCoef    =   [];
%  learning dictionary class by class
h = waitbar(0,'Training Dictionary');
for ci = 1:nClass
    cdat          =    tr_dat(:,trls==ci);
    % initialization
    D_init=Dictionary_Ini(cdat,nCOL,wayInit);
    % learning
    [Dict,mCoef,er] = CDLl2_DictLearn_general(cdat,D_init,lmd1,lmd2,nIter,isShow);
    WDict      =    [WDict Dict];
    WmCoef     =    [WmCoef;mCoef];
    WDlabels   =    [WDlabels repmat(ci,[1 size(Dict,2)])];
    % showing
    % I = displayDictionaryElementsAsImage(Dict,7,7,16,16);
%     disp(['Training dictionary of class ',num2str(ci),' out of ',num2str(nClass)]);
    waitbar(ci/nClass);
end
close(h);