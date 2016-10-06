%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);
%size(W)
%size(x)
%%% YOUR CODE HERE %%%

% 1)
%wcost = params.lambda * sum(sum(sqrt((W*x) .^ 2 + params.epsilon)));
%cecost = 0.5 * norm(W'*W*x-x,'fro') ^ 2 / params.m;
%cost = wcost + cecost;
%Wgrad = params.lambda * (W*x ./((W*x).^2 + params.epsilon)) * x' + (W*x*x'*W'*W + W*W'*W*x*x' - 2*W*x*x') / params.m;

% 2)
Y = W*x; %speed up 
wcost = params.lambda * sum(sum(sqrt(Y .^ 2 + params.epsilon))) / params.m;
%cecost = 0.5 * sum(diag((W'*Y-x)' * (W'*Y-x))) / params.m;
%norm(x,'fro') is the same with sqrt(sum(diag(x'*x)))
cecost = 0.5 * norm(W'*Y-x,'fro') ^ 2 / params.m; % use norm to speed up
%cecost = 0.5 * sum(sum((W'*Y-x).^2));
cost = wcost + cecost;
Wgrad = params.lambda * (Y ./sqrt(Y.^2 + params.epsilon)) * x' / params.m + (Y*Y'*W + W*W'*Y*x' - 2*Y*x') / params.m;


% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);