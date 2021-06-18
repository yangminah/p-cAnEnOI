addpath('/pylon5/ms5pi5p/minah/L96/RE/')
%zSPREAD = [0.05;0.05;0.25;0.45]; 
%fSPREAD = [1.1;0.7;0.8;0.9]; 
%LOCRAD = [12;12;24;36];
zSPREAD = [0.1;0.05;0.25;0.45]; 
fSPREAD = [0.8;0.7;0.8;0.9]; 
LOCRAD = [16;12;24;36];
trials = 8;

%%Create HDF5 file to save RMSE data to.
h5name = '/pylon5/ms5pi5p/minah/L96/RE/RMSE_every.h5';
if(~exist(h5name,'file'))
	h5create(h5name,'/RMSE',[4,trials,25]);
	%h5create(h5name,'/RE',[4,25,2]);
end
loc1='/pylon5/ms5pi5p/minah/L96/RE/';
%{
for j = 2 : 5
    for nn = 1 : 25
        load(sprintf('%s%02d/DATA%02d.mat',loc1,2^(j),2^(j)));
        load(sprintf('%s%02d/Networks%02d_%02d.mat',loc1,2^(j),2^(j),nn));
	XTest = dlarray(reshape(data,[1 dsize 1 n.TEST]),'SSCB');
	[z,zMean,zLogvar]=sampling(encoderNet, XTest);
	xPred = forward(decoderNet,z);
	sd = (extractdata(XTest)-extractdata(xPred)).^2;
        smm = sqrt(mean(mean(sd)));
	msm = mean(sqrt(mean(sd)));
        h5write(h5name,'/RE',[smm msm],[j-1,nn,1],[1,1,2]);
    end
end
fprintf('Done with saving reconstruction errors.\n')
%}
for kk=1:trials % 8 independent experiments for each parameter setting
    fprintf('Trial %d begins.\n',kk)
    for j = 2 : 2
        zSpread = zSPREAD(j-1);
	fSpread = fSPREAD(j-1);
	locRad = LOCRAD(j-1);
        cd(sprintf('%s%02d/%1d/',loc1,2^(j),kk));
	% The file projectAndReshapeLayer.m needs to be in the current directory when loading Networks.mat
	load(sprintf('%s%02d/vaePARAMS%02d.mat',loc1,2^(j),2^(j)));
	fprintf('\tpatchsize:%02d\n',2^(j))
     	for nn = 1 : 25
            if(~exist(sprintf('DATAc_%02d.mat',nn),'file'))
                load(sprintf('%s%02d/Networks%02d_%02d.mat',loc1,2^(j),2^(j),nn));
                cAnEnOIp
                %save to HDF5 file.
                h5write(h5name,'/RMSE',mean(sqrt(mean((Xt(:,74:end)-AM(:,74:end)).^2))),[j-1,kk,nn],[1,1,1]);
	    end
	    fprintf('Network%02d\t',nn)
	end
    end
    fprintf('\nDone with %1d:\n',kk)
end
