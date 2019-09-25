function  y = lambdafun(x, met, a, y0, y1, p)
% lambdafun       Can be used to set the forgetting factor (lambda) in RLS-DLA.
% Lambda is usually just below 1, or 1 if no forgetting at all.
% 
% special use:
%   lam = lambdafun('col');    % make figure in colors
%   lam = lambdafun('bw');     % make figure in black and white
% normal use:
%   lam = lambdafun(x, met, a, y0, y1, p);    % where
% -------------- arguments ----------
%  lam : the function value, resulting lambda
%  x   : the iteration number. It may be a fraction, i.e. x = noIt + i/L
%        where noIt is number of iterations already done, i is training
%        vector number, and L is total number of training vectors in set.
%  met : The form of the function to use
%        'P' : power        y = y1 - (y1-y0)*(1-x/a)^p,   0 <= x <= a
%        'H' : hyperbola    y = y1 - (y1-y0)*1/(x/a+1),   0 <= x
%        'E' : exponential  y = y1 - (y1-y0)*0.5^(x/a),   0 <= x
%        'L' : linear,    i.e. 'P' with p = 1
%        'Q' : quadratic, i.e. 'P' with p = 2
%        'C' : cubic,     i.e. 'P' with p = 3
%        'T' : shape is like atan function between -3 and 3
%  a   : when x=a, the maximum value, y1, is returned. 
%        or (for 'H' and 'E') when x=a, (y0+y1)/2 is returned.
%  y0  : the function value at x=0, (and x<0), default 0.99
%  y1  : the function value at x=a (and x>a) or at x=inf, default 1.00
%  p   : the power to use for met 'P', appropriate values 1: linear, 
%        2: quadratic and 3: cubic. default 1
% ------------------------------------

%----------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  xx.04.2009  KS: function made
% Ver. 1.1  07.05.2009  KS: make figure in color or black and white
% Ver. 1.2  01.09.2011  KS: added the 'T' method 
%----------------------------------------------------------------------

% This might be true, or not:
% The hyperbola method is 'theoretical' optimal in some sence when 'a' has 
% the correct value (??). Let the training set be L vectors which are
% all used in one iteration. The size of the dictionary is NxK. A good
% value of 'a' is then a = (50*K)/L. (hyperbola method only)

if (nargin == 1)  % make a figure
    mark = 'ov<*sp>hx+ovov<*spv<>*hx+ov';
    if strcmpi(x,'bw')
        col = 'kkkkkkkkkkkkkkkkkk';
        epsfile = 'lambdafun_bw.eps';
    elseif strcmpi(x,'col')
        col = 'bgrcmkybgrcmkbgrcmk';  
        epsfile = 'lambdafun_col.eps';
    else
        disp(' ');
        disp('lambdafun: first argument not as expected.');
        disp('Make figure in colors anyway.');
        disp(' ');
        col = 'bgrcmkbgrcmkbgrcmk';  
        epsfile = 'lambdafun_col.eps';
    end
    x = linspace(0,200,801); 
    y = zeros(6, numel(x));
    y(1,:) = lambdafun(x, 'L', 200, 0.99, 1.00); 
    y(2,:) = lambdafun(x, 'Q', 200, 0.99, 1.00); 
    y(3,:) = lambdafun(x, 'C', 200, 0.99, 1.00); 
    y(4,:) = lambdafun(x, 'H', 10, 0.99, 1.00); 
    y(5,:) = lambdafun(x, 'E', 20, 0.99, 1.00); 
    y(6,:) = lambdafun(x, 'T', 200, 0.99, 1.00); 
    legtxt = {'Linear, (L-200)','Quadratic, (Q-200)','Cubic, (C-200)',...
        'Hyperbola, (H-10)','Exponetial, (E-20)','atan shape, (T-200)'};
    % plot  as in dltest11
    mpnt = [200,400,600];  % markers on these points
    % col = 'rgbcmkrgbcmkrgbcmkrgbcmkb';
    figure(1);clf; hold on;
    for i = 1:6;
        h = plot(x, y(i,:), [col(i),'-']);
        set(h,'LineWidth',1.0);
        plot(x(mpnt), y(i,mpnt), [col(i),mark(i)]);
    end
    [temp,J] = sort(y(1:6,300));
    ypos = 0.9901; yinc = 0.0007;
    for j=J'
        plot([x(300),x(480)], [y(j,300),ypos], [col(j),mark(j),'-']);
        h = text(x(495), ypos, legtxt{j});
        set(h,'BackgroundColor',[1,1,1]);
        set(h,'Color',col(j));
        ypos = ypos+yinc;
    end
    
    V = [0, 200, 0.989, 1.001]; axis(V);
    % title('Different methods for assigning values to \lambda.');
    xlabel('Iteration number'); ylabel('\lambda');
    print('-f1','-depsc2',epsfile);
    y = ['lambdafun printed figure as: ',epsfile];
    disp(y);
    return;
elseif (nargin < 3)
    y = 'lambdafun: wrong number of input arguments, see help.';
    disp(y);
    return;
end
if (nargin < 4); y0 = 0.99; end;
if (nargin < 5); y1 = 1.00; end;
if (nargin < 6); p = 1; end;

y = zeros(size(x));
x = x(:);

if strcmpi(met(1),'L');
    x = 1 - x./a;
    y(:) = y1 - (y1-y0)*x;
    y(x<0) = y1;
    y(x>1) = y0;
elseif strcmpi(met(1),'Q');
    x = 1 - x./a;
    y(:) = y1 - (y1-y0)*(x.*x);
    y(x<0) = y1;
    y(x>1) = y0;
elseif strcmpi(met(1),'C');
    x = 1 - x./a;
    y(:) = y1 - (y1-y0)*(x.^3);
    y(x<0) = y1;
    y(x>1) = y0;
elseif strcmpi(met(1),'P');
    x = 1 - x./a;
    y(:) = y1 - (y1-y0)*(x.^p);
    y(x<0) = y1;      % x is -x here!
    y(x>1) = y0;
elseif strcmpi(met(1),'H');
    x = 1 + x./a;
    y(:) = y1 - (y1-y0)*1./x;
    y(x<1) = y0;  % one is added to x
elseif strcmpi(met(1),'E');
    y(:) = y1 - (y1-y0).*exp( -(log(2).*x)./a );
    y(x<0) = y0;
elseif strcmpi(met(1),'T');
    x0 = -3; 
    x1 = -x0;
    b = (y1-y0)/(atan(x1)-atan(x0));
    x = x0+((x1-x0)/a).*x;      % range set to x0 to x1 (for 0 to a)
    y(:) = (y0 - b*atan(x0)) + b*atan(x);
    y(y<y0) = y0; 
    y(y>y1) = y1;
end

return;

