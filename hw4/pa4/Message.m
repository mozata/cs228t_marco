classdef Message
	properties
		P %% Precision (aka lambda = 1/v)
		h %% Potential (aka eta = lambda mu)
	end

	methods
		function msg = Message(P,h)
			msg.P = P;
			msg.h = h;
		end
	end
end

