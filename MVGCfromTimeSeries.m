clc
clear
%% Parameters

nx = 1; % number of target ("to") variables
ny = 1; % number of source ("from") variables
nz = 0; % number of conditioning variables (default: 0)

morder = 10; % maximum model order for model order estimation
regmode = 'LWR'; % VAR model estimation regression mode 'LWR'(default) or 'OLS'

tstat     = 'F';     % statistical test for MVGC:  'F' for Granger's F-test (default) or 'chi2' for Geweke's chi2 test
alpha     = 0.05;   % significance level for significance test
mhtc      = 'FDR';  % multiple hypothesis test correction

%% import time series
prompt = 'File directory --> ';
filedirect = input(prompt,'s'); 
file = importdata(filedirect);
X = (file.data)';

%% Calculate
[numvar,numobserve,ntrials] = size(X);
[aic,bic,moaic,mobic] = tsdata_to_infocrit(X,morder); % calculate aic&bic from time series return best model order as moaic and mobic

[A,SIG,E] = tsdata_to_var(X,mobic,regmode); % fit VAR model to time series, return A:VAR coefficients matrix
% SIG:residuals covariance matrix, E:residuals time series

[G,info] = var_to_autocov(A,SIG); % calculate G:autocovariance sequence for a VAR model

F = autocov_to_pwcgc(G); % calculate pairwise-conditional time-domain multivariate Granger causalities

pval = mvgc_pval(F,mobic,numobserve,ntrials,nx,ny,nz,tstat); % p-values for sample MVGC based on theoretical asymptotic null distribution
sig = significance(pval,alpha,mhtc); % statistical significance adjusted for multiple hypotheses
F_sparse = F.*sig;

%% Show
F_sparse_inv = 1 - F_sparse;
for i = 1:numvar
    F_sparse_inv = min(min(F_sparse_inv));
end
imagesc(F_sparse_inv)
colormap gray
