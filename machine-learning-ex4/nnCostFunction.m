function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
         
z2 = [ones(m, 1) X] * Theta1';
h1 = sigmoid(z2);
z3 = [ones(m, 1) h1] * Theta2';
h2 = sigmoid(z3);
temp_y = zeros(m, size(h2,2));
for i = 1:length(y)
        temp_y(i, y(i)) = 1;
end
p1 = -log(h2) .* temp_y;
p2 = -log(ones(size(h2,1),num_labels) - h2).*(ones(size(temp_y,1),num_labels) - temp_y);
J_temp = (1/m) * sum(sum(p1 + p2, 2));
Theta1_noBias = Theta1(:, 2:size(Theta1, 2));
Theta2_noBias = Theta2(:, 2:size(Theta2, 2));
J_regular = (lambda / (2*m)) * (sum(sum(Theta1_noBias .* Theta1_noBias, 2)) + sum(sum(Theta2_noBias .* Theta2_noBias, 2)));
J = J_temp + J_regular;

h2_temp = zeros(size(y));
for i = 1:size(h2,1)
    for j = 1: num_labels
        if  h2(i,j) == max(h2(i))
            h2_temp(i) = j;
        end
    end
end
delta3 = h2 - temp_y;
delta2 = delta3 * Theta2(:, 2:end) .* sigmoidGradient(z2);
Delta = 0;
%Delta = sum(sum(delta3 .* h2, 2)) + sum(sum(delta2 .* h1, 2)); 

Delta1 = delta2' * [ones(m, 1) X];
Delta2 = delta3' * [ones(m, 1) h1];
Theta1_grad = (1/size(X,1))* Delta1;
Theta2_grad = (1/size(X,1))* Delta2;

%Delta1 = sum(sum(Delta + delta2' * X, 2));
%Delta2 = Delta1 + sum(sum(delta3' * h1));
%Theta1_grad = (1/size(X,1))* Delta1;
%Theta2_grad = (1/size(X,1))* Delta2;

% Part 3: Implement regularization with the cost function and gradients.

regular_theta1 = (lambda/m) * Theta1(:, 2:end);
regular_theta2 = (lambda/m) * Theta2(:, 2:end);
Theta1_grad = Theta1_grad + [zeros(hidden_layer_size,1) regular_theta1];
Theta2_grad = Theta2_grad + [zeros(num_labels,1) regular_theta2];
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
