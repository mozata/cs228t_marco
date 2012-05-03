function	[model z] = onlineEM(X,numGaussians,blockSize,alpha,m0,kappa0,nu0,S0);
%% Online EM for Gaussian Mixture models. Use the batch stepwise framework from 
%% Liang09, except with the MAP updates as in the slides for lecture 3.

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

%% CS228T Structured Probabilistic Models : Theoretical foundations (Spring 2012)
%% Copyright (C) 2012, Stanford University

	seed = 0;
	rand('seed',seed);
	randn('seed',seed);
	figure; 

	fullX = X;
	[numSamples numDimensions] = size(fullX);
	blocks = blockify(fullX,blockSize);
	numBlocks = numel(blocks);
	numEpochs = 1;

	disp(numBlocks);

	model = struct();
	T = numBlocks*numEpochs;

	%% step-sizes
	beta = 0.55; 
	eta = ([1:T]+2) .^ (-beta);

	%% Initialization 
	%% (1/4) INSERT CODE HERE
	N = size(X,2);
	K = numGaussians;
	phi = ones(1,K)/K;
	mu = initializeMus(blocks{1},K); % use first batch only
	Sigma = repmat(S0,[1,1,K]);
	model.phi = phi;
	model.mu = mu;
	model.Sigma = Sigma;
	% running sum of expected sufficient statistics
	rk = zeros(1,K); % sum of all responsibilities for each k
	sxk = zeros(K,N); % weighted sum of all x_i
	sSk = zeros(N,N,K); % weighted sum of all x_i x_i^T 

	
	for t=1:T
	  i = mod(t-1,numBlocks)+1; 
		X = blocks{i};
		blockSize = size(X,1);

		%% This is all the data we have seen so far
		plotModel(fullX(1:min(t*blockSize,numSamples),:),model,t);
		mov(t) = getframe;
	
		%% E-step		
		%% Compute the expected sufficient statistics of this block
		%% (2/4) INSERT CODE HERE
		M = blockSize;

		% the posterior distribution over z P(z_ik = 1 | ...) = r(i,k)
		r = zeros(M,K);
		for k=1:K
		  r(:,k) = phi(k)*mvnpdf(X,mu(k,:),Sigma(:,:,k));
		end
		r = bsxfun(@times,r,1./sum(r,2));		
		
		% expected suff. stats for prior
		rk = (1-eta(t))*rk + eta(t)*sum(r,1); % with adaptive weight
		%rk = rk + sum(r,1);

		% expected suff. stats for Gaussians
		for k=1:K
		  sxk(k,:) = (1-eta(t))*sxk(k,:) + eta(t)*sum(bsxfun(@times,r(:,k),X),1); % with adaptive rate
		  sSk(:,:,k) = (1-eta(t))*sSk(:,:,k) + eta(t)*X'*(bsxfun(@times,X,r(:,k))); % with adaptive rate
		  %sxk(k,:) = sxk(k,:) + sum(bsxfun(@times,r(:,k),X),1);
		  %sSk(:,:,k) = sSk(:,:,k) + X'*(bsxfun(@times,X,r(:,k)))
		end

		
		%% M-step
		%% Update the model parameters as in the description of
		%% stepwise EM in Liang09, but with MAP instead of MLE.
		%% Refer to slides from lecture 3 for the formulae
		%% (3/4) INSERT CODE HERE
		

		phi = ( rk + alpha - 1)/(M + sum(alpha) - K); % MAP
		%phi = rk/M; % ML

		for k=1:K
		  xk = sxk(k,:)/rk(k);

		  mu(k,:) = ( sxk(k,:) + kappa0*m0')/(rk(k) + kappa0); % MAP
		  %mu(k,:) = xk; % Ml	  

		  S = S0 + sSk(:,:,k) - sxk(k,:)'*sxk(k,:)/rk(k) + (kappa0*rk(k)/(kappa0 + rk(k)))*(xk - m0')'*(xk - m0'); % MAP
		  Sigma(:,:,k) = S/(nu0 + rk(k) + N + 2);
		  %Sigma(:,:,k) = (1/rk(k))*sSk(:,:,k) - mu(k,:)'*mu(k,:); % ML


		  % sometimes the Sigma ends up very slightly non-symmetric (1e-15 errors). correct this.
		  Sigma(:,:,k) = (Sigma(:,:,k) + Sigma(:,:,k)')/2;

		end

		model.phi = phi;
		model.mu = mu;
		model.Sigma = Sigma;
		
	end
	movie(mov,1);

	%% Calculate and return z 
	%% z(i) is the predicted label for the ith instance. 
	%% (4/4) INSERT CODE HERE

	r = zeros(numSamples,K);
	for k=1:K
	  r(:,k) = phi(k)*mvnpdf(fullX,mu(k,:),Sigma(:,:,k));
	end
	r = bsxfun(@times,r,1./sum(r,2));		
	[junk,z] = max(r,[],2);
	z = z';
	
end

function mu = initializeMus(X,numGaussians)
%% Initializes the means by sampling several sets of numGaussian 
%% points from the first batch, and picking the set of points which
%% maximizes the sum of the pairwise distances
	[numSamples numDimensions] = size(X);
	numSets = numSamples * 20;
	randPicks = ceil(rand(numSets,numGaussians) * numSamples);
	for i=1:numSets
		distances(i) = sum(pdist(X(randPicks(i,:),:)));
	end
	[val ind] = max(distances);
	mu = X(randPicks(ind,:),:);
end

function blocks = blockify(X,blockSize)
	blocks = {};
	[numSamples numDimensions] = size(X);
	numBlocks = ceil(numSamples/blockSize);
	for i=1:numBlocks
		startPos = (i-1)*blockSize+1;
		endPos = min(numSamples,i*blockSize-1);
		blocks{i} = X(startPos:endPos,:);
	end
end

function plotModel(X,gmmModel,itrNumber)
	fprintf ('.');
	numGaussians = numel(gmmModel.phi);
	[numSamples numDimensions] = size(X);
	Q = zeros(numSamples, numGaussians);
	for i=1:numSamples
		Q(i,:) = gmmModel.phi .* mvnpdf(repmat(X(i,:),numGaussians,1), gmmModel.mu, gmmModel.Sigma)';
		Q(i,:) = Q(i,:)/sum(Q(i,:));
	end

	[maxQ maxI] = max(Q');	
	plotData (numGaussians,X,maxI,sprintf('Online EM : %d',itrNumber),true);
	scatter(gmmModel.mu(:,1),gmmModel.mu(:,2),'ko','filled');
	for s=1:numGaussians
		confidenceEllipse(gmmModel.Sigma(:,:,s),gmmModel.mu(s,:),'style','k-');
	end
end

