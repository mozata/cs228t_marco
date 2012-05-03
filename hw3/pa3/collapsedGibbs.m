function P = collapsedGibbs(X,numGaussians,alpha,m0,kappa0,S0,nu0)
%% Collapsed Gibbs for the Gaussian Mixture Model
%% Inputs:
%% alpha : hyperparameter for the symmetric Dirichlet prior (1 x 1)
%% m0 : hyperparameter for the mean of all the mu (numDimensions x 1) 
%% kappa0 : hyperparameter for the precision of all the mu (1 x 1)
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
T = 50;
burnIn = 20;

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
[junk,z] = min(dists,[],2);
phi = ones(1,numGaussians)/numGaussians;
Sigma = repmat(S0,[1,1,numGaussians]);

for t=1:T
	fprintf('.');
	plotData (numGaussians,X,z,sprintf('Collapsed Gibbs %d',t),true);	
	mov(t) = getframe;
	drawnow expose;

	sxk = cell(numGaussians,1);
	sSk = cell(numGaussians,1);
	Nk = cell(numGaussians,1);
	for k=1:numGaussians
	  sxk{k} = sum(X(z == k,:));
	  sSk{k} = X(z == k,:)'*X(z==k,:);
	  Nk{k} = sum(z == k);
	end

	%% Collapsed-Gibbs phase %%
	%% (2/3) INSERT CODE HERE
	order = randperm(numSamples);
	for samp=1:numSamples
	  i = order(samp);
	  p = zeros(1,numGaussians);


	  % remove suff stats from old cluster
	  sxk{z(i)} = sxk{z(i)} - X(i,:);
	  sSk{z(i)} = sSk{z(i)} - X(i,:)'*X(i,:);
	  Nk{z(i)} = Nk{z(i)} - 1;
	  
	  for k=1:numGaussians
	    
	    % compute p(x_i | all other z, x, z_i = k)
	    % do MAP estimate for Gaussian parameters, evaluate
            % likelihood of x_i under this estimate
	    
	    muk = ( sxk{k} + kappa0*m0')/(Nk{k} + kappa0);
	    S = S0 + sSk{k} - sxk{k}'*sxk{k}/Nk{k} + (kappa0*Nk{k}/(kappa0 + Nk{k}))*(sxk{k}/Nk{k} - m0')'*(sxk{k}/Nk{k} - m0'); % MAP
	    Sigmak = S/(nu0 + Nk{k} + numDimensions + 2);
	    Sigmak = (Sigmak + Sigmak')/2;

	    p(k) = (Nk{k} + alpha/numGaussians)*mvnpdf(X(i,:),muk,Sigmak);
	    
	  end
	  p = p/sum(p);
	  
	  % now sample z_i
	  sel = rand < cumsum(p);
	  list = 1:numGaussians;
	  z(i) = min(list(sel));
	  
	  % add suff stats to new cluster
	  sxk{z(i)} = sxk{z(i)} + X(i,:);
	  sSk{z(i)} = sSk{z(i)} + X(i,:)'*X(i,:);
	  Nk{z(i)} = Nk{z(i)} + 1;

	end


	if t>burnIn
		P = P + sparse(z(R) == z(C));
	end
end
P = bsxfun(@rdivide, P, T-burnIn);

%% Plotting the mean of mean posterior %%
for s=1:numGaussians		
	%% (3/3) INSERT CODE HERE %%
	%% Calculate the posterior hyperparameters SN 
	%% (numDimensions x numDimensions) and mN (numDimensions x 1)
	%% corresponding to the s-th cluster.

	sxk{s} = sum(X(z == s,:));
	sSk{s} = X(z == s,:)'*X(z==s,:);
	Nk{s} = sum(z == s);
	
	muk = ( sxk{s} + kappa0*m0')/(Nk{s} + kappa0);
	S = S0 + sSk{s} - sxk{s}'*sxk{s}/Nk{s} + (kappa0*Nk{s}/(kappa0 + Nk{s}))*(sxk{s}/Nk{s} - m0')'*(sxk{s}/Nk{s} - m0'); % MAP
	Sigmak = S/(nu0 + Nk{s} + numDimensions + 2);
	Sigmak = (Sigmak + Sigmak')/2;

	mN = muk;
	SN = Sigmak;

	hold on; 
	confidenceEllipse(SN,mN,'style','k.');
	scatter(mN(1),mN(2),'ko','filled');
end

% movie(mov,1); % this gives me a memory error
