function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
z = X * theta;
h = sigmoid(z);

%J = 1/m * ( - log(h)' * y - log(ones(size(h, 1)) - h)' * (ones(size(y, 1)) - y)) + (lambda / (2*m)) * (theta' * theta);
J = 1/m * ((-1 * log(h)' * y) - log(ones(size(h)) - h)' * (ones(size(y)) - y)) + lambda / (2 * m) * (theta(2:size(theta, 1))' * theta(2:size(theta, 1)));

grad = (1/m) * X' * (h - y) ;
temp = theta;
temp(1) = 0;
grad = grad + (lambda/m) * temp;

grad = grad(:);

end
