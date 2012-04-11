% *** 2. logistic regression SGD ***
% choose lambda (regularization), eta (step size) that give the best results 
% (this is showing us the best possible performance of SGD)
% results in the final parameter w within the smallest batch
% objective on the subset
lambdas = 10.^(-1:0.5:2);
etas = 10.^(-7:1:-1);
obj = zeros(length(lambdas),length(etas));
subset = rand(m,1) < 1.0; nsub = sum(subset);
Asubset = A(:,subset);
ysub = y(subset);
for i=1:length(lambdas)
    for j=1:length(etas)
        lambda = lambdas(i);
        eta = etas(j);
        [w,p] = logreg_sgd(y,Asubset,lambda,eta);
        tmp = (w'*[ones(1,nsub); Asubset])';
        obj(i,j) = -lambda*sum(w.^2) + sum((ysub-1).*tmp  - log(1 + exp(-tmp)));
    end
end
lambda = 0.0;
eta = 1e-4;
[junk,idx] = max(obj(:));
[lambda_idx,eta_idx] = ind2sub(size(obj),idx);
% lambda = lambdas(lambda_idx)
% eta = etas(eta_idx)
