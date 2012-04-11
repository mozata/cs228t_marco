function [mu,var,p] = logreg_adf(y,X,lambda,S)
% Author: Marco Cusumano-Towner
% Assumed density filtering (ADF) for logistic regression with
% factored gaussian assumed density on the weights
% - uses factored gaussian prior on weights with variance 1/lambda
% - can be easily adapted to any GLM, just change the sigmoid function
% to something else

% Y is an Mx1 vector of labels (0,1)
% X is an NxM matrix of features
% LAMBDA is the regularization parameter (1/ initial variance)
% P are the one-step-ahead predictive posterior probabilities
% MU are the means of the weight Gaussian estimates over time
% VAR are the variances of the weight Gaussian estimates over time
% S is number of samples to use in Monte-Carlo approximation of posterior

n = size(X,1);
m = size(X,2);
X = [ones(1,m); X];
mu = zeros(n+1,m+1); % weights after each observation
var = zeros(n+1,m+1);
p = zeros(m,1); % one-step-ahead predicted probabilities

% means and variances of weight estimate gaussians
mu(:,1) = zeros(n+1,1);
var(:,1) = 1/lambda;

% sigmoid and normal pdf anonymous functions
sig = @(s)(1./(1+exp(-s)));
gauss = @(s,mu_,var_)( (1./sqrt(var_*2*pi)) .* exp(-(s - mu_).^2./(2*var_)) );

% random order
order = randperm(m);

for t=1:m

    i = order(t);

    % *** one-step-ahead predicted probabilities ***    
    % NOTE: to get full bayesian posterior probabilities, we would have to
    % integrate over each weight gaussian, which is intractable (KM
    % 8.4.4). We could use the "plug-in" approximation of the Bayes
    % Point (which for Gaussians is the same as the MAP estimate), but instead
    % we use Monte-Carlo approximation and sample from the posterior P(w|D_{1:t-1})
    if nargout > 2
        samp = normrnd(repmat(mu(:,t),1,S),repmat(sqrt(var(:,t)),1,S)); % samp is N x S
        p(i) = (1/S)*sum(sig(samp'*X(:,i)));
    end

    % *** observation update ***
    s_t_mu = X(:,i)'*mu(:,t);
    s_t_var = (X(:,i).^2)'*var(:,t);

    % only integrate 10 std deviations out (otherwise problems with
    % quad() ). We could use Gaussian quadrature, but this should
    % be a fine approximation.
    QUAD_LEFT = s_t_mu - 10*sqrt(s_t_var);
    QUAD_RIGHT = s_t_mu + 10*sqrt(s_t_var);

    if y(i) == 0
        z_t = quad(@(s) (1-sig(s)).*gauss(s,s_t_mu,s_t_var),QUAD_LEFT,QUAD_RIGHT);
        s_new_mu = (1/z_t)*quad(@(s) s.*(1-sig(s)).*gauss(s,s_t_mu,s_t_var),QUAD_LEFT,QUAD_RIGHT);
        s_new_var = (1/z_t)*quad(@(s) (s.^2).*(1-sig(s)).*gauss(s,s_t_mu,s_t_var),QUAD_LEFT,QUAD_RIGHT) - s_new_mu^2;
    elseif y(i) == 1
        z_t = quad(@(s) sig(s).*gauss(s,s_t_mu,s_t_var),QUAD_LEFT,QUAD_RIGHT);
        s_new_mu = (1/z_t)*quad(@(s) s.*sig(s).*gauss(s,s_t_mu,s_t_var),QUAD_LEFT,QUAD_RIGHT);
        s_new_var = (1/z_t)*quad(@(s) (s.^2).*sig(s).*gauss(s,s_t_mu,s_t_var),QUAD_LEFT,QUAD_RIGHT) - s_new_mu^2;        
    end
    
    delta_mu = s_new_mu - s_t_mu;
    delta_var = s_new_var - s_t_var;
    a = X(:,i).*var(:,t)/sum(X(:,i).^2 .* var(:,t));
    mu(:,t) = mu(:,t) + a*delta_mu;
    var(:,t) = var(:,t) + (a.^2)*delta_var;
    
    % *** do transition update  ***
    % just fixed parameters (no change, no transition variance)
    mu(:,t+1) = mu(:,t);
    var(:,t+1) = var(:,t) + 0;
    
end

mu = mu(:,end);
var = var(:,end);