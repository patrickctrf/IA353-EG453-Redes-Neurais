clear all;

% Carrega as matrizes X e S (Y, no roteiro) de entrada.
load('data.mat');

% O coeficiente de regularizacao, que varia de 2^-10 a 2^+10 com incremento
% multiplicativo de 2^2.
lambda=1;
% Temos 21 matrizes W (uma para cada coeficiente de regularizacao, todas de
% 784x10 elementos.
W = zeros(785, 10, 21);
% O vetor de bias que sera concatenado na matriz X e W para nao sei o que.
vetorExtraDe1 = ones(60000,1);

% Concatenando
X = [vetorExtraDe1, X];

% O vetor que guarda os valores dos lambdas usados
lambdasNormais = zeros(1,21);

%==========CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO=============

i=0;
while i < 21
    
    % Multiplicamos lambda por 4 == 2^2, a cada iteracao.
    lambda = 2^(i*2-14)
    lambdasNormais(i+1) = lambda;
    
    % A equacao utilizada para realizar os minimos quadrados eh dada e
    % explicada no roteiro.
    W(:,:, i+1) = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(785))^-1)*X(1:40000,:)'*S(1:40000,:);
    
    i = i+1;    
end

% W = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(785))^-1)*X(1:40000,:)'*S(1:40000,:);

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
        resultadoClassificacao = W(:,:,j)'*X(i,:)';
    
        % Verificamos se o resultado da ultima entrada de X foi correto.
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

W_refinado = zeros(785, 10, 21);

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
    lambda = 2^(i*2-14 + j*0.2)
    lambdasRefinados(j+1) = lambda;
    
    % A equacao utilizada para realizar os minimos quadrados eh dada e
    % explicada no roteiro.
    W_refinado(:,:, j+1) = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(785))^-1)*X(1:40000,:)'*S(1:40000,:);
    
    j = j+1;
end
%===FIM DO REFINADO CALCULANDO W PARA CADA COEFICIENTE DE REGULARIZACAO====



%=============TAXAS DE ERROS E ACERTOS=====================================

erros = zeros(1,21);
acertos = zeros(1,21);
taxaDeAcertos = zeros(1,21);
erroQuadratico = zeros(1,21);

% Iterando sobre os diferentes LAMBDAS. Cada "j" acessa a matriz W gerada
% por um diferente LAMBDA.
j=1;
while j<=21
    
    erros(j) = 0;
    acertos(j) = 0;
    
    % Iterando sobre as diferentes entradas  de X.
    i = 40000 + 1;
    while i<=60000

        % Pegamos a matriz W daquele LAMBDA e multiplicamos para cada
        % entrada de X (uma de cada vez, de acordo com "i"). 
        resultadoClassificacao = W_refinado(:,:,j)'*X(i,:)';
        
        % Verificamos se o resultado da ultima entrada de X foi correto.
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
        erroQuadratico(j) = erroQuadratico(j) + mean((resultadoClassificacao-S(i,:)').^2);
        
        i = i + 1;
    end
    
    
    taxaDeAcertos(j) = acertos(j)/(acertos(j)+erros(j));
    j = j +1;
end

[~,melhorResultadoErroQuadratico ] = min(erroQuadratico);
[~,melhorResultadoTaxaDeAcertos ] = max(taxaDeAcertos);


% Calculando a matriz de classificadores W definitiva, agora apenas com o
% melhor coeficiente de regularizacao Lambda encontrado e com todas as
% 60000 amostras de treinamento.
lambda = 2^((melhorResultadoErroQuadratico-1)*2-14 + (melhorResultadoErroQuadratico-1)*0.2);
W_final(:,:, j+1) = ((X'*X+lambda*eye(785))^-1)*X'*S;

dlmwrite('n175480.txt', W_final, 'precision','%17.15f');

fileID = fopen('p175480.txt','w');
fprintf(fileID, '%f\n', lambdasNormais(melhorResultadoErroQuadratico));
fclose(fileID);

fileID = fopen('Q1_175480.txt','w');
fprintf(fileID, '%f\n%f\n', lambda, lambdasNormais(melhorResultadoTaxaDeAcertosNormal));% lambdasNormais(melhorResultadoErroQuadraticoNormal), lambdasNormais(melhorResultadoTaxaDeAcertosNormal));
fclose(fileID);

% semilogx(lambdasRefinados, erroQuadratico, lambdasRefinados, taxaDeAcertos*5*10^7);
% 
% title('2-D Line Plot')
% xlabel('x')
% ylabel('cos(5x)')

%=============FIM DE TAXAS DE ERROS E ACERTOS==============================

