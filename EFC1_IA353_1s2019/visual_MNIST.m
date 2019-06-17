% 22/03/2019 - DCA/FEEC/Unicamp
% Este programa permite visualizar as imagens normalizadas da base MNIST
% Requer a dispoonibilidade do arquivo data.mat fornecido pelo professor.
% Para visualizar os dados de teste, é necessário trocar os trechos comentados.
clear all;
load('data.mat');
% load('test.mat');
[nl,nc] = size(X);
% [nl,nc] = size(Xt);
for ind = 1:nl,
    k = 1;
    for i=1:28,
        for j=28:-1:1,
            v(i,j) = X(ind,k);
%             v(i,j) = Xt(ind,k);
            k = k+1;
        end
    end
    pcolor([1:28],[1:28],v');
    colorbar;
    disp('Saída desejada')
    disp(S(ind,:));
%     disp(St(ind,:));
    disp('Digite ENTER para visualizar o próximo dígito ou CTRL-C para interromper a execução')
    pause;
end
