function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)
%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
%fprintf('numHidden = %d\n',numHidden);
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
numSamples = size(data,2);
%% forward prop
%%% YOUR CODE HERE %%%
hAct{1} = stack{1}.W * data + repmat(stack{1}.b,1,numSamples);
hAct{1} = 1 ./(1 + exp(-hAct{1}));
for l = 2:numHidden,
    hAct{l} = stack{l}.W * hAct{l-1} + repmat(stack{l}.b,1,numSamples);
    hAct{l} = 1 ./(1 + exp(-hAct{l}));
end;
out_layer = numHidden+1;
hAct{out_layer} = stack{out_layer}.W * hAct{out_layer-1} + repmat(stack{out_layer}.b,1,numSamples);
y_hat = exp(hAct{out_layer});
y_hat_sum = sum(y_hat,1);
hAct{out_layer} = bsxfun(@rdivide,y_hat,y_hat_sum);
pred_prob = hAct{out_layer};
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
y_hat = log(hAct{out_layer});
%fprintf('size(labels,1) = %d,size(labels,2) = %d\n',size(labels,1),size(labels,2));
I = sub2ind(size(y_hat),labels',1:numSamples);
cecost = -sum(y_hat(I));
%fprintf('cecost = %d\n',cecost);
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
error_term = zeros(size(hAct{out_layer})); 
error_term(I) = 1; % 10 * m
%prime_f = hAct{out_layer} .* (1 - hAct{out_layer});
%error_term = -(error_term - hAct{out_layer}) .* prime_f; % k * m
error_term = -(error_term - hAct{out_layer});
gradStack{out_layer}.W = error_term * hAct{out_layer-1}';
%fprintf('size(error_term,1) = %d,size(error_term,2) = %d\n',size(error_term,1),size(error_term,2));
gradStack{out_layer}.b = sum(error_term,2);
for l = numHidden:-1:2,
    prime_f = hAct{l} .* (1 - hAct{l});
    error_term = (stack{l+1}.W' * error_term) .* prime_f;
    gradStack{l}.W = error_term * hAct{l-1}';
    gradStack{l}.b = sum(error_term,2);
end;
prime_f = hAct{1} .* (1 - hAct{1});
error_term = (stack{2}.W' * error_term) .* prime_f; 
gradStack{1}.W = error_term * data';
gradStack{1}.b = sum(error_term,2);
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wcost = 0;
for l = 1:numHidden+1,
    wcost = wcost + 0.5 * ei.lambda * sum(stack{l}.W(:) .^ 2 );
end;
cost = cecost + wcost;
cost = cost ./ numSamples;
for  l = 1:numHidden+1,
    gradStack{l}.b = gradStack{l}.b / numSamples;
    gradStack{l}.W = gradStack{l}.W / numSamples;
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end;


%% reshape gradients into vector
[grad] = stack2params(gradStack);
%size(grad)
end