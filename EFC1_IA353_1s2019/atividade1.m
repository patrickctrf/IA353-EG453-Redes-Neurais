clear all;

% Carrega as matrizes X e S (Y, no roteiro) de entrada.
load('data.mat');

% O coeficiente de regularizacao, que varia de 2^-10 a 2^+10 com incremento
% multiplicativo de 2^2.
lambda=1;
% Temos 21 matrizes W (uma para cada coeficiente de regularizacao, todas de
% 784x10 elementos.
W = zeros(784, 10, 21);
% O vetor de bias que sera concatenado na matriz X e W para nao sei o que.
vetorExtraDe1 = ones(60000,1);

%==========CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO=============

i=0;
while i < 21
    
    % Multiplicamos lambda por 4 == 2^2, a cada iteracao.
    lambda = 2^(i*2-10)
    
    % A equacao utilizada para realizar os minimos quadrados eh dada e
    % explicada no roteiro.
    W(:,:, i+1) = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(784))^-1)*X(1:40000,:)'*S(1:40000,:);
    
    i = i+1;    
end

% W = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(784))^-1)*X(1:40000,:)'*S(1:40000,:);

%==========FIM DO CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO======



% W(:,:,1)'*X(1,:)';