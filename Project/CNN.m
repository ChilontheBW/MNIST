function y=cnn()
    path = 'E:\Users\chilo\Documents\MATLAB\';
    imgTrainPath = fullfile(path,'train.csv');
    imgTestPath = fullfile(path,'test.csv');
    imgTrain = csvread(imgTrainPath, 1, 2); % offset row by 1 for headers, and offset cols by 2 for ID and label
    imgValidation = csvread(imgTrainPath, 1, 0, [1, 0, 60000, 1]);
    test_images = csvread(imgTestPath, 1, 1);
    test_labels = csvread(imgTestPath, 1, 0, [1,0,10000,0]);    
    size(imgValidation(:,2))
    
    size(imgTrain)
    
    layers = [
        imageInputLayer([28 28 1])

        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer
    ];   

    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',4, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',30, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    net = trainNetwork(imgTrain,imgValidation(:,2),layers,options);
    
    results = classify(net,imgValidation);

    accuracy = sum(results == imgValidation)/numel(imgValidation);
    out = [imgValidation(:,1),YPred];
    outputM = ['Id','label';out];
        
    csvwrite(fullfile(path,'output.csv',outputM));
    
    y= accuracy;
end
