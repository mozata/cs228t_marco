function [w,p] = logreg_sgd(y,A,lambda,eta)
% Author: Marco Cusumano-Towner
% Logistic regression with Gaussian prior on weights (L2-regularization)
% - using stochastic gradient descent (SGD)
% - only does a single pass through the data!
% Y is the Mx1 vector of labels (0,1)
% A is the NxM matrix of features
% LAMBDA is regularization parameter
% ETA is step size (fixed)
% W are the final weight parameters after seeing all training data
% (approximation of MAP estimate)
% P are the one-step-ahead predictive posterior probabilities


n = size(A,1); % number of features
m = size(A,2); % number of examples

p = nan(m,1); % one-step-ahead predicted probabilties
w = zeros(n+1,1); % initial weight parameters, n+1 for bias term


% use step size method from pmtk3 stochgradSimple():
% etaf = @(t) 1/(1.0*t); % todo change these params...

sig = @(z)(1./(1+exp(-z)));

% for iters = 1:40
order = randperm(m);
for k=1:m
    i = order(k);
    z = w'*[1;A(:,i)];
    p(i) = sig(z);
    grad = -2*lambda*w + (y(i) - p(i))*[1;A(:,i)];
    wold = w;
    w = w + eta*grad; % maximizing objective
    if any(isnan(w))
        disp(k);
        z
        break;
    end
end
% end


end


