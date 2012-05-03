function P = gibbs(X,numGaussians,alpha,m0,V0,S0,nu0)
%% Vanilla Gibbs for the Gaussian Mixture Model
%% Inputs:
%% alpha : hyperparameter for the symmetric Dirichlet prior (1 x numGaussians)
%% m0 : hyperparameter for the mean of all the mu (numDimensions x 1) 
%% V0 : hyperparameter for the covariance of all the mu 
%%			(numDimensions x numDimensions)
%% S0 : hyperparameter that is proportional to the mean of all the Sigma 
%%			(numDimensions x numDimensions)  
%% nu0 : hyperparameter for the precision of all the Sigma (1 x 1) 
%% 
%% Outputs:
%% P : Pairwise probabilities for being in the same cluster.

%% CS228T Structured Probabilistic Models : Theoretical foundations (Spring 2012)
%% Copyright (C) 2012, Stanford University

seed = 0;
rand('seed',seed);
randn('seed',seed);
figure; 

[numSamples numDimensions] = size(X);
P = sparse(numSamples, numSamples);
[R C] = meshgrid(1:numSamples);
T = 60;%; 60;
burnIn = 30; %30;

%% Initialization %%
%% z has to be a vector of labels (numSamples x 1)
%% (1/3) INSERT CODE HERE
disp('initializing...sorry for the wait');
% using your initialization code but with more samples (the
% initializations with * 20 were sometimes bad and caused failures)
numSets = numSamples * 100;
randPicks = ceil(rand(numSets,numGaussians) * numSamples);
for i=1:numSets
  distances(i) = sum(pdist(X(randPicks(i,:),:)));
end
[val ind] = max(distances);
mu = X(randPicks(ind,:),:);
%mu = initializeMus(X,numGaussians);
dists = zeros(numSamples,numGaussians);
for k=1:numGaussians
  dists(:,k) = sum((bsxfun(@minus,X,mu(k,:))).^2,2);
end
[junk,z] = min(dists,[],2); % just for the first drawing??
phi = ones(1,numGaussians)/numGaussians;
Sigma = repmat(S0,[1,1,numGaussians]);

%figure
%scatter(X(:,1),X(:,2),10,z); hold on;
%scatter(mu(:,1),mu(:,2),50,'k*');

%% Gibbs phase %%
for t=1:T
  
	fprintf('.');
	plotData (numGaussians,X,z,sprintf('Gibbs %d',t),true);
	mov(t) = getframe;
	drawnow expose;

	%% Gibbs sampling
	%% (2/3) INSERT CODE HERE

	% 1. sample all the z given the phi, x, mu, sigma	
	r = zeros(numSamples,numGaussians);
	for k=1:numGaussians
	  r(:,k) = phi(k)*mvnpdf(X,mu(k,:),Sigma(:,:,k));
	end
	r = bsxfun(@times,r,1./sum(r,2));
	N = repmat(1:numGaussians,numSamples,1);
	Z = bsxfun(@lt,rand(numSamples,1), [cumsum(r,2)]); % checkme
	N(~Z) = numGaussians + 1;
	[junk,z] = min(N,[],2);

%	figure
%	scatter(X(:,1),X(:,2),10,z); hold on;

	% 2. sample phi given z
	alpha_new = alpha;
	for k=1:numGaussians
	  alpha_new(k) = alpha(k) + sum(z == k);
	end
	phi = drchrnd(alpha_new,1);

	% 3. sample mu | sig, x, z
	for k=1:numGaussians
	  Nk = sum(z == k);
	  xk = (sum(bsxfun(@times,z==k,X),1)/Nk)';
	  V = inv(inv(V0) + Nk*inv(Sigma(:,:,k)));
	  m = V*(inv(Sigma(:,:,k))*Nk*xk + inv(V0)*m0);
	  mu(k,:) = mvnrnd(m',V);
	end

	% 4. sample sig | mu, x, z
	for k=1:numGaussians
	  S = S0;
	  zm = bsxfun(@minus,X,mu(k,:));
	  for m=1:numSamples
	    S = S + (z(m) == k)*(zm(m,:)'*zm(m,:));
	  end
	  Nk = sum(z == k);
	  nu = nu0 + Nk;

	  sigmaModel = struct();
	  sigmaModel.Sigma = S;
	  sigmaModel.dof = nu;
	  Sigma(:,:,k) = invWishartSample(sigmaModel,1);
	end

	if t>burnIn
		P = P + sparse(z(R) == z(C));
	end
end
P = bsxfun(@rdivide, P, T-burnIn);

%% Plotting the mean of mean posterior %%
for s=1:numGaussians		
	%% (3/3) INSERT CODE HERE %%
	%% Compute the hyperparameter mN corresponding to the mean of the mean of the s-th cluster.
	%% Compute the hyperparameters SN and nuN for the variance of the s-th cluster.
	% TODO: is this what they want?

%	sxk = sum(X(z == s,:));
%	sSk = X(z == s,:)'*X(z==s,:);
%	Nk = sum(z == s);

%	muk = s
%	muk = ( sxk + kappa0*m0')/(Nk + kappa0);
%	S = S0 + sSk - sxk'*sxk/Nk + (kappa0*Nk/(kappa0 + Nk))*(sxk/Nk - m0')'*(sxk/Nk - m0'); % MAP
%	Sigmak = S/(nu0 + Nk + numDimensions + 2);
%	Sigmak = (Sigmak + Sigmak')/2;

%	mN = muk;
%	SN = Sigmak;
	
%	hold on; 
%	confidenceEllipse(SN,mN,'style','k.');
%	scatter(mN(1,:),mN(2,:),'ko','filled');
end

% movie(mov,1);
