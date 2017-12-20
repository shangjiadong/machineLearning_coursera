function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(size(X,1),1) X];
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a2 = sigmoid(X * Theta1');
a2 = [ones(size(a2, 1),1) a2];
a3 = sigmoid(a2 * Theta2'); 
h = a3;

for i = 1:size(h,1)
    for j = 1:num_labels
        if h(i,j) == max(h(i,:), [], 2)
            p(i) = j;
        end
    end
end


end
