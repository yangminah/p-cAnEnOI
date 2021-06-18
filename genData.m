%% Generate data for 1/ps s* domain_size patches.

ps=8;
loc = strcat('./');
%% Set-up parameters for the Lorenz model.
%p   struct with fields K, J, h, and F
p.K = 41;
p.J = 64;
p.h = 0.5;
p.F = 8;

ysize = p.K*p.J;
y0 = randn(ysize,1);
tspan = [0 200];
opts = odeset('RelTol',1e-3,'AbsTol',1e-6);
[T,Y] = ode45(@(t,y) RHS(t,y,p), tspan, y0, opts);

dsize = ysize/ps;
shift = dsize/2-1
Y = [Y(:,end-shift:end) Y Y(:,1:shift)];

%% Shuffle and split into training and test data.
sprintf('There are %d time steps.',length(T))
tn = length(T)*4;
%rind = randperm(tn);
n.TRAIN = ceil(.75*tn);
n.TEST = tn - n.TRAIN;
trainData = zeros(n.TRAIN,dsize);
testData = zeros(n.TEST,dsize);
for i = 1 : n.TRAIN
	trand = randperm(length(T),1);
	xrand = randperm(ysize,1);
	trainData(i,:) = Y(trand,xrand:xrand+dsize-1);
end
for i = 1 : n.TEST
	trand = randperm(length(T),1);
	xrand = randperm(ysize,1);
	testData(i,:) = Y(trand,xrand:xrand+dsize-1);
end
%trainData = Y(rind(1:n.TRAIN),:);
%testData = Y(rind(n.TRAIN+1:end),:);
save(strcat(loc,'DATA.mat'),'trainData','testData');

%% Save parameters for vae.
latentDim = ceil(1/6*dsize);
imageSize = [1 dsize 1];
save(strcat(loc,'vaePARAMS.mat'),'n','p','latentDim','imageSize','dsize');
