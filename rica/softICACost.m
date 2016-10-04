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
Wgrad = W * (2 * (W' * W * x - x)) * x' + 2 * (W * x) * (W' * W * x - x)';
cost = params.lambda * sum(sum(sqrt(W*x .^ 2 + params.epsilon))) / params.m;
tmp = (W' * W * x - x) .^ 2 / 2;
cost = cost + sum(tmp(:)) / params.m;
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);