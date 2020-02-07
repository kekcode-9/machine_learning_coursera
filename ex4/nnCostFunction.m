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
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%disp(size(Theta1));
%disp(size(Theta2));
deltaL=zeros(m,num_labels);
delta2=zeros(m,hidden_layer_size+1);
delta1=zeros(m,input_layer_size+1);
yv = zeros(m, num_labels); %make y a 5000x10 matrix
for i=1:m,
  a2=zeros(hidden_layer_size,1);
  h=zeros(num_labels,1);
  yv(i, y(i)) = 1;
  %feed forward
  a1=X(i,:)';
  a1=[1;a1];
  a2=sigmoid(Theta1*a1);
  a2=[1;a2];
  h=sigmoid(Theta2*a2);
  %a2=a2(2:size(a2,1));
  %calculate cost
  J=J-((yv(i,:)*log(h))+((1-yv(i,:))*log(1-h)));
  %backpropagation
  deltaL(i, :)=h-yv(i, :)';
  delta2(i, :)=(Theta2'*deltaL(i, :)').*a2.*(1-a2);
  Theta1_grad=Theta1_grad+delta2(i,2:hidden_layer_size+1)'*a1';
  Theta2_grad=Theta2_grad+deltaL(i,:)'*a2';
end;
disp(size(Theta1_grad));
disp(size(Theta2_grad));
regtheta1=sum(sum(Theta1(:,2:(input_layer_size+1)).^2));
regtheta2=sum(sum(Theta2(:,2:(hidden_layer_size+1)).^2));
J=(J/m)+((lambda/(2*m))*(regtheta1+regtheta2));

temp1=[zeros(size(Theta1_grad,1),1) (lambda/m).*Theta1(:,2:(input_layer_size+1))];
temp2=[zeros(size(Theta2_grad,1),1) (lambda/m).*Theta2(:,2:(hidden_layer_size+1))];
Theta1_grad=(Theta1_grad./m)+temp1;
Theta2_grad=(Theta2_grad./m)+temp2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

