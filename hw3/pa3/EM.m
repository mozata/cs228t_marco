function	[model z] = EM(X,numGaussians,alpha,m0,kappa0,nu0,S0);
%% batch MAP EM for GMM

%% Inputs:
%% X : (numSamples x numDimensions)
%% numGaussians  
%% blockSize 
%%
%% alpha : hyperparameter for the symmetric Dirichlet prior (1 x numGaussians)
%% m0 : hyperparameter for the mean of all the mu (numDimensions x 1)
%% kappa0 : hyperparameter for the precision of all the mu (1 x 1)
%% nu0 : hyperparameter for the precision of all the Sigma (1 x 1) 
%% S0 : hyperparameter that is proportional to the mean of all the Sigma 
%%			(numDimensions x numDimensions)  
%%
%% Outputs:
%% model is a struct containing: 
%% i. phi (1 x numGaussians) 
%% ii. mu (numGaussians x numDimensions)
%% iii. Sigma (numDimensions x numDimensions x numGaussians)
%% 
%% z is the vector of labels for all data items.


M = size(X,1);
N = size(X,2);
K = numGaussians;

%% Initialization
% initialize parameters pi  mu, sigma
% phi is 1 x K vector
% mu is a K x N matrix
% Sigma is a N x N x K array

seed = 1;
rand('seed',seed);
randn('seed',seed);

phi = ones(1,K)/K; % make max of prior? OR use initial mus??
mu = initializeMus(X,K);
Sigma = repmat(S0,[1,1,K]);

figure;
scatter(X(:,1),X(:,2),10); hold on;
scatter(mu(:,1),mu(:,2),50,'k*');

figure;

model = struct();
% X is M x N 
T = 30; % iterations of EM

rk = zeros(1,K); % sum of all responsibilities for each k
sxk = zeros(K,N); % weighted sum of all x_i
sSk = zeros(N,N,K); % weighted sum of all x_i x_i^T 

for t=1:T
  disp(['t=',int2str(t)]);

  %% E-step
  r = zeros(M,K);
  for k=1:K
    r(:,k) = phi(k)*mvnpdf(X,mu(k,:),Sigma(:,:,k));
  end
  r = bsxfun(@times,r,1./sum(r,2));


  figure;
  [junk,z] = max(r,[],2);
  z = z';
  scatter(X(:,1),X(:,2),10,z); hold on;
  scatter(mu(:,1),mu(:,2),50,'k*');
  
  % expected suff stats for prior
  rk = sum(r,1);
  
  % expected suff stats for Gaussians
  for k=1:K
    sxk(k,:) = sum(bsxfun(@times,r(:,k),X),1); % todo make rate
    sSk(:,:,k) =  X'*(bsxfun(@times,X,r(:,k))); % todo make rate		  
  end
  



  %% *** M-step ***

  % MAP
  phi = ( sum(r,1) + alpha - 1)/(M + sum(alpha) - K);

  % ML
%  phi = rk/M;
  
  for k=1:K
    xk = sxk(k,:)/rk(k);
    
    % MAP
    mu(k,:) = ( rk(k)*xk + kappa0*m0')/(rk(k) + kappa0);
    
    % ML
%    mu(k,:) = xk;
    
    % MAP

    S = S0 + sSk(:,:,k) - sxk(k,:)'*sxk(k,:)/rk(k) + (kappa0*rk(k)/(kappa0 + rk(k)))*(xk - m0')'*(xk - m0');
    Sigma(:,:,k) = S/(nu0 + rk(k) + N + 2);



%    zm = bsxfun(@minus,X,xk);
%    S = S0;
%    for m=1:M
%      S = S + r(m,k)*(zm(m,:)'*zm(m,:));
%    end
%    S = S + (kappa0*rk(k)/(kappa0 + rk(k)))*(xk - m0')'*(xk - m0');
%    Sigma(:,:,k) = S/(nu0 + rk(k) + N + 2);
    
    % ML
    %Sigma(:,:,k) = (1/rk(k))*sSk(:,:,k) - mu(k,:)'*mu(k,:);


    Sigma(:,:,k) = (Sigma(:,:,k) + Sigma(:,:,k)')/2; % numerical issues

  end

  
  model.phi = phi;
  model.mu = mu;
  model.Sigma = Sigma;
  
%  figure;
%  plotModel(X,model,t);

%  save('batch_EM_model.mat','model');




end

r = zeros(M,K);
for k=1:K
  r(:,k) = phi(k)*mvnpdf(X,mu(k,:),Sigma(:,:,k));
end
r = bsxfun(@times,r,1./sum(r,2));		
[junk,z] = max(r,[],2);
z = z';



