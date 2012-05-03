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
