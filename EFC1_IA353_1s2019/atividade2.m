% W: 500x10
% X: 60000x785
% V: 500x785    Montado aleatoriamente
% H: 60000x500
% f() = tanh()

% f(VxX') = H
% Saida = HxW

clear all;
close all;
clc;

% Carrega as matrizes X e S (Y, no roteiro) de entrada.
load('data.mat');
% Carrega a semente, seed (s), do gerador aleatorio.
load('seed.mat');

% O coeficiente de regularizacao, que varia de 2^-10 a 2^+10 com incremento
% multiplicativo de 2^2.
lambda=1;
% Temos 21 matrizes W (uma para cada coeficiente de regularizacao, todas de
% 500x10 elementos.
W = zeros(501, 10, 21);
% O vetor de bias que sera concatenado na matriz X e W para nao sei o que.
vetorExtraDe1 = ones(60000,1);

% Concatenando
X = [vetorExtraDe1, X];

% Ganhos iniciais para as entradas dos neuronios.
rng(s);
V = 0.2.*randn(500, 785);

% Matriz responsavel por armazenar as saidas dos neuronios.
H = tanh(X*V');
H = [vetorExtraDe1, H];

% O vetor que guarda os valores dos lambdas usados
lambdasNormais = zeros(1,21);

%==========CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO=============

i=0;
while i < 21
    
    % Multiplicamos lambda por 4 == 2^2, a cada iteracao.
    lambda = 2^(i*2-18)
    lambdasNormais(i+1) = lambda;
    
    % A equacao utilizada para realizar os minimos quadrados eh dada e
    % explicada no roteiro.
    W(:,:, i+1) = ((H(1:40000,:)'*H(1:40000,:)+lambda*eye(501))^-1)*H(1:40000,:)'*S(1:40000,:);
    
    i = i+1;    
end

% W = ((H(1:40000,:)'*H(1:40000,:)+lambda*eye(501))^-1)*H(1:40000,:)'*S(1:40000,:);

%==========FIM DO CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO======




%=============TAXAS DE ERROS E ACERTOS=====================================

erros = zeros(1,21);
acertos = zeros(1,21);
taxaDeAcertosNormal = zeros(1,21);
erroQuadraticoNormal = zeros(1,21);

% Iterando sobre os diferentes LAMBDAS. Cada "j" acessa a matriz W gerada
% por um diferente LAMBDA.
j=1;
while j<=21
    
    erros(j) = 0;
    acertos(j) = 0;
    
    % Iterando sobre as diferentes entradas  de X.
    i = 40000 + 1;
    while i<=60000
        
        % Pegamos a matriz W daquele LAMDA e multiplicamos para cada
        % entrada de X (uma de cada vez, de acordo com "i"). 
        resultadoClassificacao = W(:,:,j)'*H(i,:)';
    
        % Verificamos se o resultado da ultima entrada de H foi correto.
        [~, indiceMaxResuladoClassificacao] = max(resultadoClassificacao);
        [~, indiceMaxS] = max(S(i,:));

        if indiceMaxS == indiceMaxResuladoClassificacao
            acertos(j) = acertos(j) + 1;
        else
            erros(j) = erros(j) + 1;
        end
        
        % O erro quadratico medio eh a distancia ao quadrado media entre
        % cada item (dimensao) da saida do classificador e o respectivo
        % VETOR de referencia na MATRIZ "S".
        erroQuadraticoNormal(j) = erroQuadraticoNormal(j) + mean((resultadoClassificacao-S(i,:)').^2);
        
        i = i + 1;
    end
    
    % Estavamos somando para as 20000 amostras de validacao. Agora,
    % dividimos para ter a media.
    erroQuadraticoNormal(j) = erroQuadraticoNormal(j)/20000;
    
    taxaDeAcertosNormal(j) = acertos(j)/(acertos(j)+erros(j));
    j = j +1;
end

[~,melhorResultadoErroQuadraticoNormal ] = min(erroQuadraticoNormal);
[~,melhorResultadoTaxaDeAcertosNormal ] = max(taxaDeAcertosNormal);

% semilogx(lambdasNormais, erroQuadraticoNormal, lambdasNormais, taxaDeAcertosNormal*5*10^7)

% title('2-D Line Plot')
% xlabel('x')
% ylabel('cos(5x)')

%=============FIM DE TAXAS DE ERROS E ACERTOS==============================





%==========REFINADO CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO====

W_refinado = zeros(501, 10, 21);

% O vetor que guarda os valores dos lambdas usados
lambdasRefinados = zeros(1,21);

% Faremos uma varredura em torno do minimo encontrado. Escolheremos o valor
% do coeficiente anterior a aquele que produziu o melhor resultado e iremos
% avancando de 2^0.2, multiplicativamente, ate alcancar o coeficiente de
% classificacao seguinte ao de melhor resultado.
i = melhorResultadoErroQuadraticoNormal - 1 -1;% Lembre que o "i" comecava em zero
                                        % na primeira varredura, entao
                                        % devemos descontar "-1" novamente,
                                        % pois os indices dos vetores em
                                        % matlab comecam em "1" (nao "0") e
                                        % queremos deixar centrado no meio
                                        % o coeficiente que gerou o melhor
                                        % resultado antes de refinar.
j=0;
while j < 21
    
    % Multiplicamos lambda por 2^0.2 a cada iteracao.
    lambda = 2^(i*2-18 + j*0.2)
    lambdasRefinados(j+1) = lambda;
    
    % A equacao utilizada para realizar os minimos quadrados eh dada e
    % explicada no roteiro.
    W_refinado(:,:, j+1) = ((H(1:40000,:)'*H(1:40000,:)+lambda*eye(501))^-1)*H(1:40000,:)'*S(1:40000,:);
    
    j = j+1;
end
%===FIM DO REFINADO CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO====



%=============BUSCA REFINADA ERRO QUADRADTICO MEDIO========================

erros = zeros(1,21);
acertos = zeros(1,21);
erroQuadratico = zeros(1,21);

% Iterando sobre os diferentes LAMBDAS. Cada "j" acessa a matriz W gerada
% por um diferente LAMBDA.
j=1;
while j<=21
    
    erros(j) = 0;
    acertos(j) = 0;
    
    % Iterando sobre as diferentes entradas  de H.
    i = 40000 + 1;
    while i<=60000

        % Pegamos a matriz W daquele LAMBDA e multiplicamos para cada
        % entrada de H (uma de cada vez, de acordo com "i"). 
        resultadoClassificacao = W_refinado(:,:,j)'*H(i,:)';
        
        % Verificamos se o resultado da ultima entrada de H foi correto.
        [~, indiceMaxResuladoClassificacao] = max(resultadoClassificacao);
        [~, indiceMaxS] = max(S(i,:));
        
        % O erro quadratico medio eh a distancia ao quadrado media entre
        % cada item (dimensao) da saida do classificador e o respectivo
        % VETOR de referencia na MATRIZ "S".
        erroQuadratico(j) = erroQuadratico(j) + mean((resultadoClassificacao-S(i,:)').^2);
        
        i = i + 1;
    end
    
    % Estavamos somando para as 20000 amostras de validacao. Agora,
    % dividimos para ter a media.
    erroQuadratico(j) = erroQuadratico(j)/20000;
          
    taxaDeAcertos(j) = acertos(j)/(acertos(j)+erros(j));
    j = j +1;
end

[~,melhorResultadoErroQuadratico ] = min(erroQuadratico);

%=============FIM BUSCA REFINADA ERRO QUADRADTICO MEDIO====================



%==========REFINADO CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO====

W_refinado = zeros(501, 10, 21);

% O vetor que guarda os valores dos lambdas usados
lambdasRefinadosTaxaDeAcertos = zeros(1,21);

% Faremos uma varredura em torno do minimo encontrado. Escolheremos o valor
% do coeficiente anterior a aquele que produziu o melhor resultado e iremos
% avancando de 2^0.2, multiplicativamente, ate alcancar o coeficiente de
% classificacao seguinte ao de melhor resultado.
i = melhorResultadoTaxaDeAcertosNormal - 1 -1;% Lembre que o "i" comecava em zero
                                        % na primeira varredura, entao
                                        % devemos descontar "-1" novamente,
                                        % pois os indices dos vetores em
                                        % matlab comecam em "1" (nao "0") e
                                        % queremos deixar centrado no meio
                                        % o coeficiente que gerou o melhor
                                        % resultado antes de refinar.
j=0;
while j < 21
    
    % Multiplicamos lambda por 2^0.2 a cada iteracao.
    lambda = 2^(i*2-18 + j*0.2)
    lambdasRefinadosTaxaDeAcertos(j+1) = lambda;
    
    % A equacao utilizada para realizar os minimos quadrados eh dada e
    % explicada no roteiro.
    W_refinado(:,:, j+1) = ((H(1:40000,:)'*H(1:40000,:)+lambda*eye(501))^-1)*H(1:40000,:)'*S(1:40000,:);
    
    j = j+1;
end
%===FIM DO REFINADO CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO====



%=============BUSCA REFINADA TAXA DE ACERTOS===============================

erros = zeros(1,21);
acertos = zeros(1,21);
taxaDeAcertos = zeros(1,21);

% Iterando sobre os diferentes LAMBDAS. Cada "j" acessa a matriz W gerada
% por um diferente LAMBDA.
j=1;
while j<=21
    
    erros(j) = 0;
    acertos(j) = 0;
    
    % Iterando sobre as diferentes entradas  de H.
    i = 40000 + 1;
    while i<=60000

        % Pegamos a matriz W daquele LAMBDA e multiplicamos para cada
        % entrada de H (uma de cada vez, de acordo com "i"). 
        resultadoClassificacao = W_refinado(:,:,j)'*H(i,:)';
        
        % Verificamos se o resultado da ultima entrada de H foi correto.
        [~, indiceMaxResuladoClassificacao] = max(resultadoClassificacao);
        [~, indiceMaxS] = max(S(i,:));

        if indiceMaxS == indiceMaxResuladoClassificacao
            acertos(j) = acertos(j) + 1;
        else
            erros(j) = erros(j) + 1;
        end
        
        i = i + 1;
    end
    
    
    taxaDeAcertos(j) = acertos(j)/(acertos(j)+erros(j));
    j = j +1;
end

[~,melhorResultadoTaxaDeAcertos ] = max(taxaDeAcertos);

%=============FIM BUSCA REFINADA TAXA DE ACERTOS===========================

% Calculando a matriz de classificadores W definitiva, agora apenas com o
% melhor coeficiente de regularizacao Lambda encontrado e com todas as
% 60000 amostras de treinamento.
lambdaErroQuad = 2^((melhorResultadoErroQuadraticoNormal-1-1)*2-18 + (melhorResultadoErroQuadratico-1)*0.2);
lambdaTaxaAcertos = 2^((melhorResultadoTaxaDeAcertosNormal-1-1)*2-18 + (melhorResultadoTaxaDeAcertos-1)*0.2);
W_final = ((H'*H+lambdaTaxaAcertos*eye(501))^-1)*H'*S;


fileID = fopen('Q2_175480.txt','w');
fprintf(fileID, 'ErroQuad: %30.26f\nTaxaAcertos: %30.26f\n', lambdaErroQuad, lambdaTaxaAcertos);
fclose(fileID);



figure(1);

semilogx(lambdasNormais, erroQuadraticoNormal);

title('Sem Refinamento');
xlabel('Coeficiente de Regularização');
ylabel('Erro Quadrático Médio');

figure(2);

semilogx(lambdasNormais, taxaDeAcertosNormal);

title('Sem Refinamento');
xlabel('Coeficiente de Regularização');
ylabel('Taxa de Acertos');



figure(3);

semilogx(lambdasRefinadosTaxaDeAcertos, taxaDeAcertos);

title('Com Refinamento');
xlabel('Coeficiente de Regularização');
ylabel('Taxa de Acertos');

figure(4);

semilogx(lambdasRefinados, erroQuadratico);

title('Com Refinamento');
xlabel('Coeficiente de Regularização');
ylabel('Erro Quadrático Médio');