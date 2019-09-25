% ========================================================================
% Simplified Fisher Discriminative Dictionary Learning (FDDL), Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
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

demo_simplified_FDDL.m   Digit recognition demo on USPS database with raw feature

Utilities: 
	folder of Simplified FDDL functions, including
Eigenface_f:  
	function of computing PCA Projection Matrix
Simplified_FDDL:     
	main function of FDDL
Comp_CentSparseCoding3_largedata:   
	function of computing coding coefficient in S-FDDL
CDLl2_DictLearn_general:     
	function of dictionary learning
Dictionary_Ini:      
	function of initializing dictionary
SolveDALM_M:         
	function of sparse coding based on Allan Yang's code (L1Solvers)

% Please refer to the following papers

% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang, "Fisher Discrimination Dictionary Learning for Sparse Representation," 
% in Proc. 13th IEEE International Conference on Computer Vision (ICCV), pp. 543?550, Barcelona, Spain, 2011.

% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang, "Sparse representation based Fisher discrimination dictionary learning for image classification," 
% International Journal of Computer Vision (IJCV), 2014.

%-------------------------------------------------------------------------
Contact: yangmengpolyu@gmail.com; cslzhang@comp.polyu.edu.hk

%---------------------note------------------------------------
We modify the code of DALM based on the previous work of "A. Y. Yang, A. Ganesh, Z. H. Zhou, S. S. Sastry, and Y. Ma. 
Fast  l1-minimization algorithms and application in robust face recognition, UC Berkeley, Tech. Rep" 
%---------------------note------------------------------------
