function ratings = batchEPTrueSkill(G,W,blockSize)
%% Performs a single sweep of EP over the games data in blocks,
%% performing multiple iterations of message passing per block of data
%% to approximately calculate the mean and the variance of each 
%% player's skill.

%% NOTE : All messages passed are in the form of Gaussian precisions and potentials.
%%		Even if a node is non Gaussian internally, it will both receive and send
%%		its messages in this format.

%% Refer the handout for details of the model

%% Inputs:
%%		G (M x 2) : G(i,1) is the winner of the i-th game, 
%%						and G(i,2) is the loser of the i-th game
%%		W (N x 1) : cell-array with W{i} the name of the i-th player


	M = size(W,1);       % number of players
	N = size(G,1);       % number of games in 2011 season 

	pv = 0.5;            % prior skill variance (prior mean is always 0)
	blocks = blockify(G,blockSize);
	numBlocks = numel(blocks);
	
	% struct to store the auxiliary variables. 
	% see initAux and updateAux for details
	aux = initAux(W,M,pv);

	%% Initializing the messages from the game nodes to the skill nodes
	mgs = Message(zeros(N,2), zeros(N,2));
	% mgs.P(i,1) is the precision for message from game i to first player of game i
	% mgs.P(i,2) is the precision from game i to second player in game
	% mgs.h(i,1), mgs.h(i,2) similar
	
	for i=1:numBlocks	
		aux = updateAux(aux,blocks{i},i,blockSize);

		for iter=1:3
			fprintf('.');
			
			%% Message from skill nodes to game nodes
			msg = msgSkillToGame(mgs,aux);
				
			%% Message from game nodes to performance nodes
			mgp = msgGameToPerformance(msg,aux);
			
			%% Message from performance nodes to game nodes
			mpg = msgPerformanceToGame(mgp,aux);
			
			%% Message from game nodes to skill nodes
			mini_mgs = msgGameToSkill(mpg,msg,aux);
 
			mgs.P(aux.blockBegin:aux.blockEnd,:) = mini_mgs.P;
			mgs.h(aux.blockBegin:aux.blockEnd,:) = mini_mgs.h;

		end
	end
	
	aux.G = G;
	ratings = getRatings(mgs,aux);
end

function ratings = getRatings(mgs,aux)
	for p=1:aux.M
		Ps(p) = 1/aux.pv + sum(mgs.P(aux.G==p)); 
		hs(p) = sum(mgs.h(aux.G==p));
	end
	ratings = hs./Ps;
end

function	msg = msgSkillToGame(mgs,aux) 
%% INSERT CODE HERE %%
%	allN = size(mgs.P,1);
N = aux.N;

Pgs = mgs.P(aux.blockBegin:aux.blockEnd,:);
hgs = mgs.h(aux.blockBegin:aux.blockEnd,:);

	% (1) compute skill node beliefs 
	P = repmat((1/aux.pv),aux.M,1);
	h = zeros(aux.M,1); % prior mu is 0
	for g=1:N
	  p1 = aux.G(g,1); % player 1
	  p2 = aux.G(g,2); % player 2
	  P(p1) = P(p1) + Pgs(g,1);
	  P(p2) = P(p1) + Pgs(g,2);
	  h(p1) = h(p1) + hgs(g,1);
	  h(p2) = h(p2) + hgs(g,2);
	end

	% (2) compute skill to game messages
	Psg = zeros(N,2); % Psg(i,1) message from first player to game i
	% Psg(i,2) message from second player to game i
	hsg = zeros(N,2);
	for g=1:N
	  p1 = aux.G(g,1);
	  p2 = aux.G(g,2);
	  Psg(g,1) = P(p1) - Pgs(g,1);
	  Psg(g,2) = P(p2) - Pgs(g,2);
	  hsg(g,1) = h(p1) - hgs(g,1);
	  hsg(g,2) = h(p2) - hgs(g,2);
	end
	
	msg = Message(Psg,hsg);
end

function mgp = msgGameToPerformance(msg,aux)
	%% INSERT CODE HERE %%
	% (3) compute game to performance messages
	% Remember that player 1 always wins the way we store data (doesn't this only matter in msgPerformanceToGame?)
	N = aux.N;

	Pgp = zeros(N,1);
	hgp = zeros(N,1);
	for g=1:N
	  P1 = msg.P(g,1); % skill to game messages
	  P2 = msg.P(g,2);
	  h1 = msg.h(g,1);
	  h2 = msg.h(g,2);
	  
	  mu1 = h1/P1; % skill to game messages in (mu,v) form
	  mu2 = h2/P2;
	  v1 = 1/P1;
	  v2 = 1/P2;

	  v = 1 + v1 + v2;
	  Pgp(g) = 1/v;
	  mu = mu1 - mu2;
	  hgp(g) = mu*Pgp(g);
	end
	
	mgp = Message(Pgp,hgp);
end

function mpg = msgPerformanceToGame(mgp,aux)	
	%% Useful functions 
	psi = inline('normpdf(x)./normcdf(x)');
	lambda = inline('(normpdf(x)./normcdf(x)).*( (normpdf(x)./normcdf(x)) + x)');

	lambda2 = inline('normpdf(x)./(1-normcdf(x))');
	delta2 = inline('(normpdf(x)./(1-normcdf(x))).*(normpdf(x)./(1-normcdf(x)) - x)');



	%% INSERT CODE HERE %%
	N = aux.N;

	Ppg = zeros(N,1);
	hpg = zeros(N,1);
	for g=1:N

	  % posterior over difference variables (project)
	  var_gp = 1/mgp.P(g);
	  sig_gp = sqrt(var_gp);
	  mu_gp = mgp.h(g)/mgp.P(g);

	  mu_p = 1*mu_gp + sig_gp*psi(1*mu_gp/sig_gp); % y_g = +1 
	  var_p = var_gp*(1 - lambda(1*mu_gp/sig_gp));
	  mu_p2 = mu_gp + sig_gp*lambda2(-mu_gp/sig_gp);
	  var_p2 = var_gp*(1 - delta2(-mu_gp/sig_gp));

	  Pp = 1/var_p;
	  hp = mu_p*Pp;

	  % performance to game messages
	  Ppg(g) = Pp - mgp.P(g);
	  hpg(g) = hp - mgp.h(g);
	end

	mpg = Message(Ppg,hpg);
end

function mgs = msgGameToSkill(mpg,msg,aux)
	%% INSERT CODE HERE %%
	% (6) compute game to skills messages
	N = aux.N;
	Pgs = zeros(N,2);
	hgs = zeros(N,2);
	
	for g=1:N
	  Psg1 = msg.P(g,1); % previous messages from skill to game
	  Psg2 = msg.P(g,2);
	  hsg1 = msg.h(g,1);
	  hsg2 = msg.h(g,2);
	  mu_sg1 = hsg1/Psg1;
	  mu_sg2 = hsg2/Psg2;
	  v_sg1 = 1/Psg1;
	  v_sg2 = 1/Psg2;

	  Ppg = mpg.P(g); % previous message from performance to game
	  hpg = mpg.h(g);
	  v_pg = 1/Ppg;
	  mu_pg = hpg/Ppg;
	  
	  v_gs1 = 1 + v_pg + v_sg2; % upward messages
	  v_gs2 = 1 + v_pg + v_sg1;
	  mu_gs1 = mu_pg + mu_sg2;
	  mu_gs2 = mu_pg - mu_sg1;
	  
	  
	  Pgs(g,1) = 1/v_gs1;
	  Pgs(g,2) = 1/v_gs2;
	  hgs(g,1) = mu_gs1*Pgs(g,1);
	  hgs(g,2) = mu_gs2*Pgs(g,2);

	end
  
	mgs = Message(Pgs,hgs);
end

function aux = initAux(W,M,pv)
	aux = struct();
	aux.W = W;
	aux.M = M;
	aux.pv = pv;
end

function aux = updateAux(aux,block,i,blockSize)
	aux.G = block;
	aux.N = size(aux.G,1);
	aux.blockBegin = (i-1)*blockSize + 1;
	aux.blockEnd = aux.blockBegin + aux.N - 1;
	aux.players = unique(aux.G(:));
end


function blocks = blockify(X,blockSize)
	blocks = {};
	[numSamples numDimensions] = size(X);
	numBlocks = ceil(numSamples/blockSize);
	for i=1:numBlocks
		startPos = (i-1)*blockSize+1;
		endPos = min(numSamples,i*blockSize);
		blocks{i} = X(startPos:endPos,:);
	end
end


