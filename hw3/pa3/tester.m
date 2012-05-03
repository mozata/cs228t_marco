%% CS228T Structured Probabilistic Models : Theoretical foundations (Spring 2012)
%% Copyright (C) 2012, Stanford University

%% Read startup.m for startup instructions 
%% IMPORTANT : You may need to modify it. 
fprintf ('CS228T Structured Probabilistic Models : Theoretical foundations (Spring 2012)\n');
fprintf ('Programming Assignment 3\n\n');


numSamples = 3000;
numGaussians = 4;
numDimensions = 2;

[X y] = dataGen(numSamples,numGaussians,numDimensions);
plotData (numGaussians,X,y,'Ground truth');

maxError = 0.3;

do_batch_em = false;
do_online_em=false; %%true;
do_gibbs = true;
do_collapsed_gibbs = false;

if do_batch_em
	alpha = ones(1,numGaussians)*10;
	m0 = zeros(numDimensions,1);
	kappa0 = 0;
	S0 = eye(numDimensions,numDimensions) * 100;
	nu0 = 20;

	[model z] = EM(X,numGaussians,alpha,m0,kappa0,nu0,S0);

	%% Plot results
	plotData(numGaussians,X,z,'Batch EM Results');
	hold on; 
	scatter(model.mu(:,1),model.mu(:,2),'ko','filled');
	for s=1:numGaussians
		confidenceEllipse(model.Sigma(:,:,s),model.mu(s,:),'style','k-');
	end

	confmat = confusionmat(z,y);
	[bsNrmError bsRndError] = evaluateClustering(X,y,z);
	printTestResults(bsNrmError, maxError, 'Batch EM');
	confmat
end

if do_online_em
	alpha = ones(1,numGaussians)*10;
	m0 = zeros(numDimensions,1);
	kappa0 = 0;
	S0 = eye(numDimensions,numDimensions) * 100;
	nu0 = 20;
	blockSize = 100;
	[model z] = onlineEM(X,numGaussians,blockSize,alpha,m0,kappa0,nu0,S0);

	%% Plot results
	plotData(numGaussians,X,z,'Online EM Results');
	hold on; 
	scatter(model.mu(:,1),model.mu(:,2),'ko','filled');
	for s=1:numGaussians
	  confidenceEllipse(model.Sigma(:,:,s),model.mu(s,:),'style','k-');
	end

	confmat = confusionmat(z,y);
	[bsNrmError bsRndError] = evaluateClustering(X,y,z);
	printTestResults(bsNrmError, maxError, 'Online EM');
	confmat
end

if do_gibbs 
	%% See comments inside gibbs.m for explanation of arguments 
	alpha = ones(1,numGaussians)/numGaussians; % ??
	%alpha = ones(1,numGaussians)*10;
	m0 = zeros(numDimensions,1);
	V0 = eye(numDimensions,numDimensions) * 20;
	S0 = eye(numDimensions,numDimensions) * 20;
	nu0 = 50;
	P = gibbs(X,numGaussians,alpha,m0,V0,S0,nu0);
	
	[gibbsNrmError gibbsRndError] = evaluateClustering(X,y,P);
	printTestResults(gibbsNrmError, maxError, 'Gibbs');
end

if do_collapsed_gibbs
	%% See comments inside collapsedGibbs.m for explanation of arguments 
	m0 = zeros(numDimensions,1);
	S0 = eye(numDimensions,numDimensions) * 20;
	nu0 = 50;
	alpha = 30;
	kappa0 = 5;
	
	P = collapsedGibbs(X,numGaussians,alpha,m0,kappa0,S0,nu0);
	[cgNrmError cgRndRrror] = evaluateClustering(X,y,P);
	printTestResults(cgNrmError, maxError, 'Collapsed Gibbs');
end
