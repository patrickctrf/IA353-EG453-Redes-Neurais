clear all;
load('data.mat');

lambda=1;
W = zeros(21, 784, 10);

i=0;
while i < 21
    
    lambda = 2^(i*2-10)
    
    W(i+1,:,:) = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(784))^-1)*X(1:40000,:)'*S(1:40000,:);
    
    i = i+1;    
end

% W = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(784))^-1)*X(1:40000,:)'*S(1:40000,:);