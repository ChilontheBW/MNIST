% Kayla Bachler & Chris Otey
% Mini-Project Fashion MNIST
% 12-10-2018

function y=cnn()
    path = 'E:\Users\chilo\Documents\MATLAB\';
    imds = imageDatastore(fullfile(path,'trainImg'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsTest = imageDatastore(fullfile(path,'testImg'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
    label = 60001:70000;
 
    imdsTest.Labels =string(label);
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
        'Verbose', false, ... 
        'Plots','training-progress', ...
        'ExecutionEnvironment', 'parallel');
    
    net = trainNetwork(imdsTrain,layers,options);
    
    YPred = classify(net,imdsValidation); % classifies each row of the data in net into a label
    testResults_cat = classify(net,imdsTest)
    testResults = grp2idx(testResults_cat)
    size(testResults); % 10000x1 vector
    YValidation = imdsValidation.Labels;
    results = zeros(10000,2);
    for x = 1:10000
        results(x,:) = [imdsTest.Labels(x), testResults(x)];
    end

    TT = array2table(results,'VariableNames',{'Id' 'Label'});
    writetable(TT,fullfile(path,'results.csv'));
    
    accuracy = sum(YPred == YValidation)/numel(YValidation)
end