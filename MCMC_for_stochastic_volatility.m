% MCMC for stochastic volatility

% NPQ $2019.11.14$
% https://newportquant.com/

clear
ask = readtable('EURUSD_GMT23_weekly.csv');

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
ylabel('Volatility h_t');

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
hi = V;    
Mdl = fitlm(lagmatrix(log(hi),1),log(hi));
alphai = Mdl.Coefficients{:,1}';
Sigmavi = nanvar(Mdl.Residuals.Raw);

% --- proposing distribution for ht (reflecting random walk)
delta_hp = 0.008;
range_hp = 5;    % proposed h must be within a range from 0 to 5*var(rtn)

% --- MCMC 
nmcmc = 200000;
beta_mcmc = nan(nmcmc,1);
h_mcmc = nan(n,nmcmc);
alpha_mcmc = nan(nmcmc,length(alpha0));
Sigmav_mcmc = nan(nmcmc,1);
count_accept = 0;
rng(1);
tic
for ii=1:nmcmc
    % --- Gibbs sampling: beta
    rtn_new = rtn./sqrt(hi);  % reformat return series
    x = 1./sqrt(hi);
    V_beta = 1/(x'*x + 1/Sigma_beta0);
    E_beta = V_beta*(beta0/Sigma_beta0+x'*rtn_new);
    betai = normrnd(E_beta,sqrt(V_beta));
    
%     % --- Griddy Gibbs sampling: ht
%     ngg = 400;  % number of grid size
%     for jj=1:n
%         if jj==1
%             loghb1 = alphai(2)^2*(log(hi(2))-alphai(1)/(1-alphai(2))) + alphai(1)/(1-alphai(2));
%         else
%             loghb1 = log(hi(jj-1));
%         end
%         if jj==n
%             loghf1 = alphai(1)+alphai(2)*(alphai(1)+alphai(2)*log(hi(n-1)));
%         else
%             loghf1 = log(hi(jj+1));
%         end
%         mut = (alphai(1)*(1-alphai(2))+alphai(1)*(loghf1+loghb1))/(1+alphai(2)^2);
%         % --- span of h for the grid
%         hmin = min(hi(jj),V(jj));
%         hmax = max(hi(jj),V(jj));
%         hspan = linspace(0.6*hmin,1.4*hmax,ngg);
%         % - probability
%         p_ht = hspan.^(-1.5).*exp(-(rtn(jj)-betai)^2./(2*hspan)) ...
%             .*exp(-(log(hspan)-mut).^2/(2*Sigmavi/(1+alphai(2)^2)));
%         % - normalize probability
%         p_ht = p_ht/trapz(hspan,p_ht);
%         % - cdf
%         c_ht = cumtrapz(hspan,p_ht);
%         % - draw sample
%         [~,ia,~]=unique(c_ht);
%         hi(jj) = interp1(c_ht(ia),hspan(ia),rand);
%     end
    
     % --- Metropolis sampling: ht
     % - propose new ht
     % - with lognormal
%      var_hp = ones(size(hi))*0.0002;
%      hp =  lognrnd(log(hi.^2./sqrt(var_hp+hi.^2)), sqrt(log(var_hp./hi.^2+1)) );
     % - with reflective mirror random walk     
     hp = abs(unifrnd(hi-delta_hp,hi+delta_hp));    % uniform distribution to propose a new h
     while any(hp<=0 | hp>range_hp*var(rtn))    % ensure hp is in the range from 0 t 5*var(rtn)
         hp = abs(hp);
         hp = min([hp,2*range_hp*var(rtn)-hp],[],2);
     end
     logh0 = alphai(2)^2*(log(hi(2))-alphai(1)/(1-alphai(2))) + alphai(1)/(1-alphai(2));
     loghn1 = alphai(1)+alphai(2)*(alphai(1)+alphai(2)*log(hi(n-1)));
     loghf1 = [log(hi(2:end)); loghn1];      % log of one-step forward ht
     loghb1 = [logh0; log(hi(1:end-1))];     % log of one-step backward ht
     mut = (alphai(1)*(1-alphai(2))+alphai(1)*(loghf1+loghb1))/(1+alphai(2)^2);
     logr = -1.5*(log(hp)-log(hi)) - (rtn-betai).^2.*(1./hp-1./hi)/2 - ((log(hp)-mut).^2-(log(hi)-mut).^2)/(2*Sigmavi/(1+alphai(2)^2));
     % - accept or not
     idxi = log(rand(n,1))<logr;
     hi(idxi) = hp(idxi);
     count_accept = count_accept + nnz(idxi);

    % --- Gibbs sampling: alpha
    zt = [ones(n-1,1),log(hi(1:end-1))];
    V_alpha = inv( inv(Sigma_alpha0) + zt'*zt/Sigmavi);
    E_alpha = V_alpha*(inv(Sigma_alpha0)*alpha0' + zt'*log(hi(2:end))/Sigmavi);
    alphai = mvnrnd(E_alpha,V_alpha);
    
    % --- Gibbs sampling: Sigmav
    SSR = sum((log(hi(2:end))-zt*alphai').^2);
    % Alternative way to get SSR by OLS
%     alpha_ols = inv(zt'*zt)*(zt'*log(hi(2:end)));             % coefficent by OLS
%     SSR = sum((log(hi(2:end))-zt*alpha_ols).^2);            % SSR of OLS
    Sigmavi = 1/random('Gamma',(nu0+n-1)/2,2/(nu0*Sigmav0^2+SSR));

    % collect result
    beta_mcmc(ii) = betai;
    h_mcmc(:,ii) = hi;
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
nburnin = 140000;
figure('position',[355   320   800   200]);
hold on; box on;
ub = mean(h_mcmc(:,nburnin:end),2)+2*std(h_mcmc(:,nburnin:end),[],2);
lb  = mean(h_mcmc(:,nburnin:end),2)-2*std(h_mcmc(:,nburnin:end),[],2);
patch_x = [1:length(t), length(t):-1:1];
patch_y = [lb;flip(ub)];
patch(patch_x,patch_y,'r','EdgeColor','none','FaceAlpha',0.3);
plot(1:length(t),mean(h_mcmc(:,nburnin:end),2));%,'linewidth',1) 
ylims = ylim;
ylabel('Stochastic volatility');
idx = find(month(t)==1 & day(t)<=7 & mod(year(t),2)==0);
set(gca,'xtick',idx);
set(gca,'XTickLabel',year(t(idx)));
set(gca,'xlim',[1,length(t)]);
set(gca,'ylim',[0,ylims(2)]);

%%  distributions of parameters
param_mcmc = [beta_mcmc(nburnin:end),alpha_mcmc(nburnin:end,:),Sigmav_mcmc(nburnin:end)];
array2table([mean(param_mcmc); std(param_mcmc)],'VariableNames',{'beta','alpha1','alpha2','sigmav2'},'RowNames',{'mean','std'})
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