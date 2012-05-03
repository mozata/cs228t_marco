function [post_mu, post_var] = gibbsTrueSkill(G,W,ns,burn_in)
%% Inputs:
%%		G (M x 2) : G(i,1) is the winner of the i-th game, 
%%						and G(i,2) is the loser of the i-th game
%%		W (N x 1) : cell-array with W{i} the name of the i-th player

%% ns: number of samples
%% burn_in: number of initial sampels to discard


M = size(W,1);       % number of players
N = size(G,1);       % number of games in 2011 season 
pv = 0.5;            % prior skill variance (prior mean is always 0)

% all games won by first player in list
y = ones(N,1);

% cumulative second moments x_i^2 (for calculating variance of posteriors)
cum_second_mom = zeros(M,1);
% cumulative sum (for calculating mean and variance of posteriors)
cum_sum = zeros(M,1);

T = zeros(N,1);
w = zeros(M,1);

% effective number of games played by each player ( 1/v + N_i in my pdf)
eff_num_games = zeros(M,1);
for m=1:M
  eff_num_games(m) = 1/pv + sum(G(:) == m);
end

won = cell(M,1);
lost = cell(M,1);
for m=1:M
  won{m} = (G(:,1) == m);
  lost{m} = (G(:,2) == m);
end

for samp = 1:(burn_in + ns)
  

%  disp(['samp ',int2str(samp)]);
  % sample the T_n (we can do this in a block)
  mu = w(G(:,1)) - w(G(:,2));
  var = ones(N,1);
  T = samp_trunc_norm(mu,var);
%  for n=1:N
%    mu = w(G(n,1)) - w(G(n,2));
%    var = 1;
%    T(n) = samp_trunc_norm(mu,var,0,inf);
%  end  

  % sample the w_i
  for m=1:M
    mu = (1/(eff_num_games(m)))*(sum(w(G(won{m},2)) + T(won{m}) ) + sum(w(G(lost{m},1)) - T(lost{m})) );
    var = 1/eff_num_games(m);
    w(m) = normrnd(mu,sqrt(var));
  end

  if samp > burn_in
    % accumulate statistics
    cum_sum = cum_sum + w;
    cum_second_mom = cum_second_mom + w.^2;
  end
  

end

% posterior means and variances
post_mu = (1/ns)*cum_sum;
post_var = (1/ns)*cum_second_mom - post_mu.^2;

end

function out = samp_trunc_norm(mu,var)
% samples from [0, inf) truncated normal
% sample a bunch at once
sigma = sqrt(var);
sampleSize = 1;

N = length(mu);

a = normcdf(-mu./sigma);
u = a + (1-a).*rand(N,1);
out = mu + sigma.*(sqrt(2)*erfinv(2*u-1));

% copied from http://www.mathworks.com/matlabcentral/fileexchange/7309
%  Alex Bar Guy  &  Alexander Podgaetsky
%PHIl = normcdf((a-mu)/sigma);
%PHIr = normcdf((b-mu)/sigma);

%out = mu + sigma*( sqrt(2)*erfinv(2*(PHIl+(PHIr-PHIl)*rand(sampleSize))-1) );
end

