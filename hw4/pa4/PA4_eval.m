load data;

%[gibbs_post_mu, gibbs_post_var] = gibbsTrueSkill(G,W,1000,100);
ratings = batchEPTrueSkill(G,W,100); 
[sorted_ratings ind] = sort(ratings,'descend');

fprintf('\n\n');
for i=1:5
	fprintf('%d %3.2f %s\n', i, ratings(ind(i)), W{ind(i)});
end

err = norm(ratings-soln_ratings);
if (err > 1e-5)
	fprintf('FAILED.\n')
else
	fprintf('PASSED!\n');
end

%% INSERT CODE HERE %%
%% Generate plots as in assignment %%

figure;
%scatter(soln_ratings,gibbs_post_mu','b'); hold on;
scatter(soln_ratings,ratings','r'); hold on;
plot([0;2],[0;2],'g');
xlabel('soln_ratings');
legend('gibbs','my EP');
