% MCMC for stochastic volatility
% Companion code for blog post
% https://newportquant.com/stochastic-volatility-by-markov-chain-monte-carlo/

% NPQ $2019.11.14$

clear
ask = readtable('EURUSD_GMT23_daily.csv');

t = ask.Time;
cl = ask.close;

%% return and correlation test
rtn = double((cl - lagmatrix(cl,1))./lagmatrix(cl,1));
rtn = log(1+rtn)*100;   % log return in percentage

% - remove NaN
t = t(2:end);
cl = cl(2:end);
rtn = rtn(2:end);

n = size(rtn,1);    % total number of data

figure('position',[355   320   800   400]);
subplot(2,1,1);
plot(t,cl)
ylabel('Exchange rate');
subplot(2,1,2);
plot(t,rtn);
ylabel('Return')

% --- serial correlation check on return
figure('position',[355   320   800   400]);
plotcorrstat(t,rtn,30,1:30)
subplot(2,2,1);
ylabel('Return');

figure('position',[355   320   800   400]);
plotcorrstat(t,rtn.^2,30,1:30)
subplot(2,2,1);
ylabel('Return^2');

%% Model by GARCH(1,1)
Mdl = arima(0,0,0);
Mdl.Variance = garch(1,1);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl,rtn);
[E,V,logL] = infer(EstMdl,rtn);
res = E./sqrt(V);
% [h,pValue,stat,cValue] = lbqtest(res,'lags',12)
% [h,pValue,stat,cValue] = lbqtest(res.^2,'lags',12)
% % Note: lbqtest on both res and res.^2 do not reject the model; so model is
% % sufficient

% --- serial correlation check on residuals
figure('position',[355   320   800   400]);
plotcorrstat(t,res,30,1:30)
subplot(2,2,1);
ylabel('\epsilon_t');

figure('position',[355   320   800   400]);
plotcorrstat(t,res.^2,30,1:30)
subplot(2,2,1);
ylabel('\epsilon_t^2');

%% plot volatility
figure('position',[355   320   800   200]);
plot(t,V)
set(gca,'XMinorTick','on');
set(gca,'YMinorTick','on');
ylabel('GARCH Volatility h_t');

%% MCMC for Stochastic volotility
% --- prior parameters
% --- for beta (normal)
beta0 = 0;          % mean
Sigma_beta0 = 0.25;    % variance
% - for alpha's (normal)
alpha0 = [0,0.2];   % mean
Sigma_alpha0 = diag([0.4,0.4]);   % covariance
% - for sigma^2_v
nu0 = 1;
Sigmav0 = 0.1;

% --- initial values using GARCH(1,1) model, and least square fit of log(ht0)
betai = EstMdl.Constant;
loghi = log(V);
Mdl = fitlm(lagmatrix(loghi,1),loghi);
alphai = Mdl.Coefficients{:,1}';
Sigmavi = nanvar(Mdl.Residuals.Raw);

% --- proposing distribution for log(ht
sigma_loghp = 0.1;

% --- MCMC
nmcmc = 10000;
beta_mcmc = nan(nmcmc,1);
logh_mcmc = nan(n,nmcmc);
alpha_mcmc = nan(nmcmc,length(alpha0));
Sigmav_mcmc = nan(nmcmc,1);
count_accept = 0;
rng(1);
tic
for ii=1:nmcmc
    % --- Gibbs sampling: beta
    rtn_new = rtn./sqrt(exp(loghi));  % reformat return series
    x = 1./sqrt(exp(loghi));
    V_beta = 1/(x'*x + 1/Sigma_beta0);
    E_beta = V_beta*(beta0/Sigma_beta0+x'*rtn_new);
    betai = normrnd(E_beta,sqrt(V_beta));
    
    % --- Metropolis sampling: ht
    logh0 = alphai(2)^2*(loghi(2)-alphai(1)/(1-alphai(2))) + alphai(1)/(1-alphai(2));
    loghn1 = alphai(1)+alphai(2)*(alphai(1)+alphai(2)*loghi(n-1));
    loghf1 = [loghi(2:end); loghn1];      % log of one-step forward ht
    loghb1 = [logh0; loghi(1:end-1)];     % log of one-step backward ht
    % - propose new ht
    loghp = normrnd(loghi,sigma_loghp);
    % - check log ratio of the posterior probability
    logr = log(normpdf(loghp, [ones(n,1),loghb1]*alphai',sqrt(Sigmavi))) + ...
        log(normpdf(loghf1,[ones(n,1),loghp] *alphai',sqrt(Sigmavi))) + ...
        log(normpdf(rtn-betai,0,sqrt(exp(loghp)))) - ...
        log(normpdf(loghi, [ones(n,1),loghb1]*alphai',sqrt(Sigmavi))) - ...
        log(normpdf(loghf1,[ones(n,1),loghi] *alphai',sqrt(Sigmavi))) - ...
        log(normpdf(rtn-betai,0,sqrt(exp(loghi))));
    % - accept or not
    idxi = log(rand(n,1))<logr;
    loghi(idxi) = loghp(idxi);
    count_accept = count_accept + nnz(idxi);
    
    % --- Gibbs sampling: alpha
    zt = [ones(n-1,1),loghi(1:end-1)];
    V_alpha = inv( inv(Sigma_alpha0) + zt'*zt/Sigmavi);
    E_alpha = V_alpha*(inv(Sigma_alpha0)*alpha0' + zt'*loghi(2:end)/Sigmavi);
    alphai = mvnrnd(E_alpha,V_alpha);
    
    % --- Gibbs sampling: Sigmav
    SSR = sum((loghi(2:end)-zt*alphai').^2);
    % Alternative way to get SSR by OLS
    %      alpha_ols = inv(zt'*zt)*(zt'*loghi(2:end));             % coefficent by OLS
    %      SSR = sum((loghi(2:end)-zt*alpha_ols).^2);            % SSR of OLS
    Sigmavi = 1/random('Gamma',(nu0+n-1)/2,2/(nu0*Sigmav0^2+SSR));
    
    % collect result
    beta_mcmc(ii) = betai;
    logh_mcmc(:,ii) = loghi;
    alpha_mcmc(ii,:) = alphai;
    Sigmav_mcmc(ii) = Sigmavi;
end
toc

%% plot MCMC result to show burnin
figure('position',[355   320   800   400]);
subplot(4,1,1);
plot(beta_mcmc);
ylabel('\beta');
subplot(4,1,2);
plot(alpha_mcmc(:,1))
ylabel('\alpha_1');
subplot(4,1,3);
plot(alpha_mcmc(:,2))
ylabel('\alpha_2');
subplot(4,1,4);
plot(Sigmav_mcmc);
ylabel('\sigma_v^2');
xlabel('# of iterations');

%% ht after burnin
h_mcmc = exp(logh_mcmc);
nburnin = 4000;
lb = quantile(h_mcmc(:,nburnin+1:end),0.025,2);   % 2.5% quantile
ub = quantile(h_mcmc(:,nburnin+1:end),0.975,2);   % 97.5% quantile
patch_x = [1:length(t), length(t):-1:1];
patch_y = [lb;flip(ub)];
figure('position',[355   320   800   300]);
hold on; box on;
plot(1:length(t),V,'linewidth',1)
plot(1:length(t),mean(h_mcmc(:,nburnin+1:end),2),'linewidth',1)
patch(patch_x,patch_y,'m','EdgeColor','none','FaceAlpha',0.1);
legend({'GARCH','SV','95% interquantile'},'Location','best');
ylims = ylim;
ylabel('Stochastic volatility');
idx = find(month(t)==1 & mod(year(t),2)==0);    % get the 1st day of the year
idx( month(t(idx-1)) == 1) = [];
set(gca,'xtick',idx);
set(gca,'XTickLabel',year(t(idx)));
set(gca,'xlim',[1,length(t)]);
set(gca,'ylim',[0,3]);

%%  distributions of parameters
param_mcmc = [beta_mcmc(nburnin+1:end),alpha_mcmc(nburnin+1:end,:),Sigmav_mcmc(nburnin+1:end)];
E_param = mean(param_mcmc);
V_param = var(param_mcmc);
fprintf('mean and std of parameters are\n');
array2table([E_param; sqrt(V_param)],'VariableNames',{'beta','alpha1','alpha2','sigmav2'},'RowNames',{'Mean','SE'})

% --- autocorrelation of parameters
corr_lags = 1000;
figure
title_str = {'\beta','\alpha_1','\alpha_2','\sigma_v^2'};
for ii=1:4
    subplot(2,2,ii)
    [acf,acf_lags,acf_bounds] = autocorr(param_mcmc(:,ii),corr_lags);
    plot(acf_lags,acf);%,'Marker','none');
    hold on; box on;
    plot(acf_lags,repmat(acf_bounds',length(acf),1),'r');
    set(gca,'TickLength',[0.04,0.04]);
    set(gca,'YMinorTick','on','XMinorTick','on');
    ylabel('ACF');
    xlabel('Lag');
    title(title_str{ii})
end

% --- MCMC standard error by effective sample size
[ESS,Sigma] = multiESS(param_mcmc);    % book assume no correlation between thetas 
ESS = round(ESS);
% MCMCSE = sqrt(diag(cov(param_mcmc))'/ESS);    % useing covariance of mcmc
MCMCSE = sqrt(diag(Sigma)'/ESS);    % useing covariance from mutliESS
fprintf('Effective sample size is %d\n',ESS);
fprintf('   Approximated MCMC standard errors are [%f,%f,%f,%f]\n',MCMCSE);

% --- plot
figure
[S,AX,BigAx,H,HAx] = plotmatrix(param_mcmc);
set(AX,'xminortick','on','yminortick','on','ticklength',[0.04,0.04]);
ylabel(AX(1,1),'\beta');
ylabel(AX(2,1),'\alpha_1');
ylabel(AX(3,1),'\alpha_2');
ylabel(AX(4,1),'\sigma_v^2');
xlabel(AX(end,1),'\beta');
xlabel(AX(end,2),'\alpha_1');
xlabel(AX(end,3),'\alpha_2');
xlabel(AX(end,4),'\sigma_v^2');
set(H,'EdgeColor','none')
