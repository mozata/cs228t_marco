% logistic regression with gaussian prior on weights using
% stochastic gradient descent

clear; close all;

load('stocip-matlab/spamassasin.mat','A','y');

sig = @(z)(1./(1+exp(-z)));

n = size(A,1); % number of features
m = size(A,2); % number of examples

p = nan(m,1); % one-step-lookahead predicted probabilties
w = zeros(m,1); % initial weight parameters
lambda = 1.0; % regularization weight (TODO determine this)

for i=1:m

    p(i) = sig(w'*[1;A(:,i)]);
    grad = -2*lambda*w + (y(i) - p(i))*[1;A(:,i)];

end