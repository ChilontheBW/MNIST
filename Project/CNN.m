function y=cnn()
    path = 'E:\Users\chilo\Documents\MATLAB\';
    imds = imageDatastore(fullfile(path,'trainImg'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsTest = imageDatastore(fullfile(path,'testImg'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
    
    numTrainFiles = 5500;
    [imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
    
    layers = [
        imageInputLayer([28 28 1])
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,14,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,56,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        convolution2dLayer(3,112,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,224,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,14,'Padding','same')
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(10)
        softmaxLayer

        softmaxLayer
        classificationLayer
    ];   

    options = trainingOptions('sgdm', ...
        'MaxEpochs', 4, ...
        'MiniBatchSize', 128, ...
        'Verbose', false, ... 
        'Plots','training-progress', ...
        'ExecutionEnvironment', 'parallel');
    
    net = trainNetwork(imdsTrain,layers,options);
    
    YPred = classify(net,imdsValidation)
    testResults = classify(net,imdsTest)
    testResults(1)
    YValidation = imdsValidation.Labels
    results = zeros(10000,2);
    for x = 1:10000
        results(x,:) = [x+60000, testResults(x)];
    end
    csvwrite(fullfile(path,'results.csv'),results,1,0)
    accuracy = sum(YPred == YValidation)/numel(YValidation)
    
    y= accuracy;
end
