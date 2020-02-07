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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h=sigmoid(X*theta);
l1=((y'*log(h))+((1-y)'*log(1-h)));
l2=-l1./m;
J=l2;
s=X'*(h-y);
grad=s./m;
%you can do the part below . it is for gradient descent
%alpha=0.01;
%temp=zeros(2 + 1, 1);
%for i=1:10
% for j=1:2
%    s=sum((h-y).*X(:,j));
%    grad(j)=s./m;
%    temp(j)=theta(j)-((alpha/m)*s);
%   end;
%   theta=temp;
%end;
% =============================================================

end;
