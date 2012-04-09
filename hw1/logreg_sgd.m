function [w,p] = logreg_sgd(y,A,lambda,eta)
% logistic regression with gaussian prior on weights using
% stochastic gradient descent

sig = @(z)(1./(1+exp(-z)));

n = size(A,1); % number of features
m = size(A,2); % number of examples

p = nan(m,1); % one-step-lookahead predicted probabilties
w = zeros(n+1,1); % initial weight parameters

% even though we have a Gaussian prior, this is not really Bayesian
% because we aren't integrating over P(w), we're just using the MAP
% estimate for w at each iteration (?) is the issue that Gaussian
% is not a conjugate prior for logistic function?

% use step size method from pmtk3 stochgradSimple():
% etaf = @(t) 1/(1.0*t); % todo change these params...

% for iters = 1:40
order = randperm(m);
for k=1:m
    i = order(k);
    p(i) = sig(w'*[1;A(:,i)]);
    grad = -2*lambda*w + (y(i) - p(i))*[1;A(:,i)];
    w = w - eta*grad;

end
% end


end


