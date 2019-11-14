function varargout = plotcorrstat(varargin)
% PLOTCORRSTAT plots correlation statistics.
%
%   PLOTCORRSTAT(X,Y,ACF_LAGS,LBQ_LAGS)
%   HA = PLOTCORRSTAT(X,Y,ACF_LAGS,LBQ_LAGS) returns the handle vector for
%   the four subplots
%
%   X and Y: vectors of data. Y, such as the residuals of a regression will be correlated.
%   ACF_LAGS: lag value for ACF and PACF
%   LBQ_LAGS: vector of lags for LBQ-test

%   NPQ $2018/02/15$

x = varargin{1};
y = varargin{2};
corr_lags = varargin{3};
lbq_lags = varargin{4};
    
[acf,acf_lags,acf_bounds] = autocorr(y,corr_lags);
[pacf,pacf_lags,pacf_bounds] = parcorr(y,corr_lags);
[h,pValue,stat,cValue] = lbqtest(y,'lags',lbq_lags);

% --- remove lag=0 for acf and pacf
acf = acf(2:end);
acf_lags = acf_lags(2:end);
pacf = pacf(2:end);
pacf_lags = pacf_lags(2:end);

% --- plot
clf(gcf)
ha(1) = subplot(2,2,1);
plot(x,y);
xlabel('x'); ylabel('y');

ha(2) = subplot(2,2,2);
hold on; box on;
stem(acf_lags,acf,'marker','none')
plot(acf_lags([1,end]),repmat(acf_bounds',2,1),'r--')
xlabel('Lag'); ylabel('ACF')
%set(gca,'ylim',[min([-acf_bounds;acf]),max([acf_bounds;acf])]);

ha(4) = subplot(2,2,4);
hold on; box on;
stem(pacf_lags,pacf,'marker','none')
plot(pacf_lags([1,end]),repmat(pacf_bounds',2,1),'r--')
xlabel('Lag'); ylabel('PACF')
% set(gca,'ylim',[min([-pacf_bounds;pacf]),max([pacf_bounds;pacf])]);

ha(3) = subplot(2,2,3);
hold on; box on;
plot(lbq_lags,pValue,'o');
plot(lbq_lags([1,end]),[0.05,0.05],'--');
xlabel('Lag'); ylabel('LBQ test p-value');

linkaxes(ha([2,4]),'x');
set(ha,'ticklength',[0.03,0.03]);
set(ha,'xminortick','on','yminortick','on');

if nargout == 1
    varargout{1} = ha;
end