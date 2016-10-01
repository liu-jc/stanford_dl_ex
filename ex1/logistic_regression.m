function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
  h = 1 ./ (1 + exp(-theta' * X));
  %fprintf('size(h,1) = %d, size(h,2) = %d\n',size(h,1),size(h,2));
  %fprintf('size(y,1) = %d, size(y,2) = %d\n',size(y,1),size(y,2));
  for i = 1:m,
      f = f - (log(h(i)) * y(i) + log(1-h(i)) * (1 - y(i)));
  end;
  for j = 1:size(theta),
      for i = 1:m,
          g(j) = g(j) + X(j,i) * (h(i) - y(i));
      end;
  end;
  %f = -(log(h) * y' + (log(1 - h)) * (1 - y)');
  %fprintf('size(x,1) = %d , size(x,2) = %d\n',size(X,1),size(X,2));
  %fprintf('size(h-y,1) = %d, size(h-y,2) = %d\n',size(h-y,1),size(h-y,2));
  %g = X * (h - y)';
%%% YOUR CODE HERE %%%
