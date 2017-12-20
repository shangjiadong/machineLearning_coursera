function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

z = X * theta;
%J = 1/m * sum( -1 * (log(sigmoid(z))' * y -  (log(ones(size(z)) - sigmoid(z)))' * (ones(size(y)) - y)));
h = sigmoid(z);
J = 1/m * ((-1 * log(h)' * y) - log(ones(size(h)) - h)' * (ones(size(y)) - y));
grad = 1/m * ( X' * (sigmoid(z) - y));

end
