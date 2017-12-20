function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

z = X * theta;
h = sigmoid(z);
J = 1/m * ((-1 * log(h)' * y) - log(ones(size(h)) - h)' * (ones(size(y)) - y)) + lambda / (2 * m) * (theta(2:size(theta, 1))' * theta(2:size(theta, 1)));
grad = 1/m * ( X' * (sigmoid(z) - y)) + (lambda / m) * theta;
grad(1) = 1/m * ( X(:, 1)' * (sigmoid(z) - y));

end
