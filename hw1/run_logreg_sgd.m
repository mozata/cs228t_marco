clear; close all;

load('data/stocip-matlab/spamassassin.mat','A','y');
A = A(1:10000,:);

% choose lambda, eta hyperparameters (this seems illegal)
lambdas = 10.^(-1:0.5:1);
etas = 10.^(-5:1:-1);
acc = zeros(length(lambdas),length(etas));
for i=1:length(lambdas)
    for j=1:length(etas)
        [w,p] = logreg_sgd(y,A,lambdas(i),etas(j));
        acc(i,j) = sum( y == (p > 0.5));
   end
end
[junk,idx] = max(acc(:));
[lambda_idx,eta_idx] = ind2sub(size(acc),idx);
lambda = lambdas(lambda_idx);
eta = etas(eta_idx);

[w,p] = logreg_sgd(y,A,lambda,eta);

% confusion matrix
t = 0.5;
a = sum(y == 0 & p <= t);
b  = sum(y == 1 & p <= t);
c = sum(y == 0 & p > t);
d = sum(y == 1 & p > t);
disp([a b; c d]);

save('data/logreg_sgd.mat','w','p','lambda','eta');

% 3) treat first 3000 cases as training data, remaining 3034 as
% test data
trainsets = [100, 1000, 3000];
Atest = A(:,3001:end);
ytest = y(3001:end);
ntest = size(Atest,2)
ptest = cell(3,1);
sig = @(z)(1./(1+exp(-z)));
for k=1:3
    Atrain = A(:,1:trainsets(k));
    w = logreg_sgd(y(1:trainsets(k)),Atrain,lambda,eta);
    ptest{k} = sig(w'*[ones(1,size(Atest,2)); Atest] )';
end

% ROC curves
figure;
ts = 0:0.01:1;
for k=1:3
    subplot(1,3,k);
    tpr = zeros(length(ts),1);
    fpr = zeros(length(ts),1);
    for i=1:length(ts)
        t = ts(i);
        ypred = ptest{k} > t;
        tpr(i) = sum(and(ytest == 1, ypred == 1))/sum(ytest == 1);
        fpr(i) = sum(and(ytest == 0, ypred == 0))/sum(ytest == 0);
    end
    plot(fpr,tpr);
end

% compare to batch 