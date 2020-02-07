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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=sigmoid(X*theta);
l1=((y'*log(h))+((1-y)'*log(1-h)));
l2=-l1./m;
l3=(lambda/(2*m))*sum(theta(2:size(theta,1),:).^2);
J=l2+l3;
s=X'*(h-y);
temp=[0; (lambda).*theta(2:size(theta,1),:)];
s=(s./m)+(temp./m);
grad=s;
%grad=fprintf('%.4f\n',grad(1:size(theta,1)));

% =============================================================


