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
