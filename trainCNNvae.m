loc = './';
load(strcat(loc,'DATA.mat'));
load(strcat(loc,'vaePARAMS.mat'));

XTrain = dlarray(reshape(trainData,[1 dsize 1 n.TRAIN]),'SSCB');
XTest = dlarray(reshape(testData,[1 dsize 1 n.TEST]),'SSCB');
clear trainData testData
CNN_VAE
executionEnvironment = "auto";
numTrainImages = size(XTrain,4);
numEpochs = 3000;
miniBatchSize = 3500; % maxes out around 10GB
lr = 1e-3;
numIterations = floor(numTrainImages/miniBatchSize);
iteration = 0;

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];
for epoch = 1:numEpochs
    tic;
    for i = 1:numIterations
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XBatch = XTrain(:,:,:,idx);
        XBatch = dlarray(single(XBatch), 'SSCB');

        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);
        end

        [infGrad, genGrad] = dlfeval(...
            @modelGradients, encoderNet, decoderNet, XBatch);

        [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderNet.Learnables, ...
                genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
                infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
    end
    elapsedTime = toc;

    [z, zMean, zLogvar] = sampling(encoderNet, XTest);
    xPred = forward(decoderNet, z);
    elbo = ELBOloss(XTest, xPred, zMean, zLogvar);
    disp("Epoch : "+epoch+" Test ELBO loss = "+gather(extractdata(elbo))+...
        ". Time taken for epoch = "+ elapsedTime + "s")
    save(strcat(loc,'Networks.mat'),'encoderNet','decoderNet')
end
