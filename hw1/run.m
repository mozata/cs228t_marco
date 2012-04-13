% CS 228T Spring 2012: HW 1
% Author: Marco Cusumano-Towner
% run this script to generate all figures
% assumes 'data' directory with 'stocip-matlab/spamassassin.mat' inside

clear; close all;

load('data/stocip-matlab/spamassassin.mat','A','y');
m = size(A,2);

% randomize data order
new_order = randperm(m);
A = A(1:10000,new_order);
y = y(new_order);

n = size(A,1);

lambda = 1e-6;
eta0 = 10;
t0 = 10000;
etaf = @(t) eta0*t0/(t+t0);

sig = @(z)(1./(1+exp(-z)));

% *** SGD one-step-ahead predictive performance (confusion matrix) ***
% use step size method from pmtk3 stochgradSimple():
% etaf = @(t) eta0*t0/(t+t0);
[w,p] = logreg_sgd(y,A,lambda,etaf);
t = 0.5;
a = sum(y == 0 & p <= t);
b  = sum(y == 1 & p <= t);
c = sum(y == 0 & p > t);
d = sum(y == 1 & p > t);
disp([a b; c d]);


% *** ADF one-step-ahead predictive performance (confusion matrix) ***
[mu,var,padf] = logreg_adf(y,A,lambda,100); % 100 is number of MC samples
t = 0.5;
a = sum(y == 0 & padf <= t);
b  = sum(y == 1 & padf <= t);
c = sum(y == 0 & padf > t);
d = sum(y == 1 & padf > t);
disp([a b; c d]);

% *** get SGD and ADF ROC curves for different training sets ***
trainsets = [100,1000,3000];
p_sgd = cell(3,1);
p_adf = cell(3,1);
p_adf_var = cell(3,1);

ytest = y(3001:m);
Atest = A(:,3001:m);
ntest = length(ytest);

for i=1:3
    
    ytrain = y(1:trainsets(i));
    Atrain = A(:,1:trainsets(i));

    % SGD: get weight point estimate from SGD after 100,1000,3000 train samples
    % get posterior probabilities from point weight estimate
    w = logreg_sgd(ytrain,Atrain,lambda,etaf);
    p_sgd{i} = sig([ones(1,ntest); Atest]'*w );

    % generate ROC curves for SGD
    figure(1);
    ts = 0:0.001:1;
    subplot(1,3,i);
    tpr = zeros(length(ts),1);
    fpr = zeros(length(ts),1);
    for j=1:length(ts)
        ypred = p_sgd{i} > ts(j);
        tpr(j) = sum(and(ytest == 1, ypred == 1))/sum(ytest == 1);
        fpr(j) = sum(and(ytest == 0, ypred == 0))/sum(ytest == 0);
    end
    plot(fpr,tpr); 
    title(['SGD ROC curve ntrain=',int2str(length(ytrain))]);


    % ADF: get weight posteriors after 100,1000,3000 train samples
    % weight posteriors: mu, var
    [mu,var] = logreg_adf(ytrain,Atrain,lambda);

    % sample from weight posterior to approximate P(y_i | x_i, data) for test i
    % (as in KM 8.4.4.1)
    p_adf{i} = zeros(ntest,1);
    p_adf_var{i} = zeros(ntest,1);
    S = 1000; % number of samples
    samp = normrnd(repmat(mu,1,S),repmat(sqrt(var),1,S)); % samp is N x S
    for j=1:ntest % test set
        p_adf{i}(j) = (1/S)*sum(sig(samp'*[1; Atest(:,j)])); % mu = p
        p_adf_var{i}(j) = p_adf{i}(j)*(1-p_adf{i}(j)); % var = p(1-p)
    end

    % generate ROC curves for ADF
    figure(2);
    ts = 0:0.001:1;
    subplot(1,3,i);
    tpr = zeros(length(ts),1);
    fpr = zeros(length(ts),1);
    for j=1:length(ts)
        ypred = p_adf{i} > ts(j);
        tpr(j) = sum(and(ytest == 1, ypred == 1))/sum(ytest == 1);
        fpr(j) = sum(and(ytest == 0, ypred == 0))/sum(ytest == 0);
    end
    plot(fpr,tpr);
    title(['ADF ROC curve ntrain=',int2str(length(ytrain))]);

    % ADF: plots with error bars on estimate
    figure(4);

    % D_0
    subplot(2,3,i);
    [p_adf_sorted,sorted_order] = sort(p_adf{i}(ytest == 0),'descend');
    tmp = p_adf_var{i}(ytest == 0);
    p_adf_var_sorted = tmp(sorted_order);
    errorbar(1:length(p_adf_sorted),p_adf_sorted,sqrt(p_adf_var_sorted));
    title(['D_0, ntrain=',int2str(length(ytrain))]);
    
    % D_1
    subplot(2,3,i+3);
    [p_adf_sorted,sorted_order] = sort(p_adf{i}(ytest == 1)','descend');
    tmp = p_adf_var{i}(ytest == 1);
    p_adf_var_sorted = tmp(sorted_order);
    errorbar(1:length(p_adf_sorted),p_adf_sorted,sqrt(p_adf_var_sorted));
    title(['D_1, ntrain=',int2str(length(ytrain))]);

end

% get batch confusion matrix and ROC curve (training on 3000 samples)
[model, X, lambdaVec, opt] = logregFit(A(:,1:3000)', y(1:3000), 'regType', 'l2','lambda',lambda);
[y_hat, p_batch] = logregPredict(model, Atest');
t = 0.5;
a = sum(ytest == 0 & p_batch <= t);
b  = sum(ytest == 1 & p_batch <= t);
c = sum(ytest == 0 & p_batch > t);
d = sum(ytest == 1 & p_batch > t);
disp([a b; c d]);
figure(3);
ts = 0:0.001:1;
tpr = zeros(length(ts),1);
fpr = zeros(length(ts),1);
for j=1:length(ts)
    ypred = p_batch > ts(j);
    tpr(j) = sum(and(ytest == 1, ypred == 1))/sum(ytest == 1);
    fpr(j) = sum(and(ytest == 0, ypred == 0))/sum(ytest == 0);
end
plot(fpr,tpr);
title('batch ROC curve, nTrain=3000');


% P(weight | training data) <---- distribution
% P(y_i | training data, x_i) = integral_over_w P(y_i | weight, x_i) P(weight | train)
% take 100 samples of w, then compute P(y_i | w, x_i) (probability)