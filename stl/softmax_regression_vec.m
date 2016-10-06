function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  y_hat = [exp(theta' * X)]; % k * m
  y_hat_sum = sum(y_hat,1);% 1 * m
  p_y = bsxfun(@rdivide,y_hat,y_hat_sum);
  
  ind_mat = zeros(num_classes,m);
  I = sub2ind(size(ind_mat),y,1:size(ind_mat,2));
  ind_mat(I) = 1;
  %f = -sum(sum(ind_mat * log(p_y)'));
  f = -sum(log(p_y(I))); % p_y(I) 1*m
  g = -X * (ind_mat - p_y)';
  g=g(:); % make gradient a vector for minFunc

