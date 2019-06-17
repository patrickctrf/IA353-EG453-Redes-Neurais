Data = load('data.mat');
DataField = fieldnames(Data);
dlmwrite('data.txt', Data.(DataField{1}));

Data = load('test.mat');
DataField = fieldnames(Data);
dlmwrite('test.txt', Data.(DataField{1}));