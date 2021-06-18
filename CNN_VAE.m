encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer([1 3], 3, 'Padding','same', 'Stride', 1, 'Name', 'conv1a')
    eluLayer('Name','elu1a')
    convolution2dLayer([1 3], 9, 'Padding','same', 'Stride', 1, 'Name', 'conv1b')
    eluLayer('Name','elu1b')
    convolution2dLayer([1 3], 27, 'Padding','same', 'Stride', 1, 'Name', 'conv1c')
    eluLayer('Name','elu1c')
    maxPooling2dLayer([1 2],'Stride',[1 2],'Name','pool1') % Drops to dsize/2
    convolution2dLayer([1 3], 27, 'Padding','same', 'Stride', 1, 'Name', 'conv2a')
    eluLayer('Name','elu2a')
    convolution2dLayer([1 3], 27, 'Padding','same', 'Stride', 1, 'Name', 'conv2b')
    eluLayer('Name','elu2b')
    maxPooling2dLayer([1 2],'Stride',[1 2],'Name','pool2') % Drops to dsize/4
    convolution2dLayer([1 3], 27, 'Padding','same', 'Stride', 1, 'Name', 'conv3a')
    eluLayer('Name','elu3a')
    convolution2dLayer([1 3], 27, 'Padding','same', 'Stride', 1, 'Name', 'conv3b')
    eluLayer('Name','elu3b')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')
    ]);

decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    projectAndReshapeLayer([1 dsize/4 27],latentDim,'proj');
    eluLayer('Name','elu0')
    transposedConv2dLayer([1 3], 27, 'Cropping', 'same', 'Stride', [1 2], 'Name', 'transpose1a') % doubles to 1312
    eluLayer('Name','elu1a')
    transposedConv2dLayer([1 3], 27, 'Cropping', 'same', 'Stride', 1, 'Name', 'transpose1b') % Still at 1312
    eluLayer('Name','elu1b')
    transposedConv2dLayer([1 3], 9, 'Cropping', 'same', 'Stride', [1 2], 'Name', 'transpose2') % doubles to 2624
    eluLayer('Name','elu2a')
    transposedConv2dLayer([1 3], 9, 'Cropping', 'same', 'Stride', 1, 'Name', 'transpose2b') % still at 2624
    eluLayer('Name','elu2b')
    transposedConv2dLayer([1 3], 1, 'Cropping', 'same', 'Stride', 1, 'Name', 'transpose2c') % still at 2624
    ]);

encoderNet = dlnetwork(encoderLG);
decoderNet = dlnetwork(decoderLG);
