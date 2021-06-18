%%
%zSpread, fSpread, locrad, and loc1  should've been called in params.m
%loc1 is in the patch directory. 
ysize=p.J*p.K;
dratio=ysize/dsize; % # of patches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Configure dynamics and observing system
dtObs = 0.2; % Observation window; 0.2 = one day
Nt = 73*5; % Number of assimilation cycles
dxObs = 4; % Observe every dxObs point
No = ysize/dxObs; % Number of obs per cycle
obsErr = sqrt(0.5); % Obs error std

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% locRad, fSpread, and zSpread can be set in the Driver.m file instead of below
Ne = 100; % Ensemble size
rind = randi([0:1],[1,Ne]);
dshift = rind .* floor(dsize/Ne*(1:Ne)) + (1 - rind) .*ceil(dsize/Ne*(1:Ne));

% Set up localization
x = [0:ysize/2 -(ysize/2)+1:-1]';
loc = exp(-.5*(x/locRad).^2);
clear x

%% Allocate space for results
FM = zeros(ysize,Nt); % Forecast mean
AM = FM; % Analysis mean

%% Get reference data and obs
rng('shuffle') % ensure different initial seeds for each run
[~,Y] = ode45(@(t,y) RHS(t,y,p),[0 linspace(9,9+dtObs*(Nt-1),Nt)],...
            randn(ysize,1));
Xt = Y(2:end,:)';
clear Y
Y = Xt(1:dxObs:end,:); % Observations
Y = Y + obsErr*randn(size(Y)); % Add obs error

%% Initialize ensemble mean
[~,tmp] = ode45(@(t,y) RHS(t,y,p),[0 3 9],randn(p.J*p.K,1));
XM = tmp(3,:)';clear tmp
X = zeros(ysize, Ne);
%% cAnEnOI
for ii=1:Nt
    % Forecast mean
    FM(:,ii) = XM;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Construct analogsReformat and decode
    % Create copies of XM with different shifts.
    % We will reuse X later to store the ensemble members.
    % X = zeros(ysize, Ne); MOVED outside loop.
    for i = 1 : Ne
        %X(:,i) = circshift(XM, dshift*(i-1));
        X(:,i) = circshift(XM, dshift(i));
    end
    % Reshape to get patches.
    x = dlarray(reshape(X,[1 dsize 1 Ne*dratio]),'SSCB');
    % Encode.
    [~, zMean, ~] = sampling(encoderNet, x);
    zMean = extractdata(zMean);
    % Add noise.
    z = dlarray(reshape(zMean + zSpread*randn(size(zMean)),...
        [1 1 size(zMean)]),'SSCB');
    % Decode and reshape patches to get Ne ensemble members.
    X = squeeze(reshape(extractdata(forward(decoderNet,z)),...
        [ysize Ne]));
    % Shift in opposite direction.
    for i = 1 : Ne
        %X(:,i) = circshift(X(:,i), -dshift*(i-1));
        X(:,i) = circshift(X(:,i), -dshift(i));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    A = bsxfun(@plus,-mean(X,2),X);
    A = double(A*(fSpread/(sqrt(Ne-1)*std(A(:))))); % scaled ens pert matrix
    % Serial assimilation
    for jj=1:No
        indX = 1 + dxObs*(jj-1);
        % Obs mean
        ym = XM(indX);
        % Scaled obs perturbation matrix
        V = A(indX,:);
        s2 = V*(V'); g2 = obsErr^2;
        WHB = 1/(s2 + g2 + sqrt(s2*g2 + g2^2));
        XM = XM + circshift(loc,[indX-1 0]).*(A*(V'))*(Y(jj,ii)-ym)*(1/(s2 + g2));
        A = A - WHB*bsxfun(@times,circshift(loc,[indX-1 0]),A*(V')*V);
    end
    % Analysis mean
    AM(:,ii) = XM;
    % forecast mean
    [~,sol] = ode45(@(t,y) RHS(t,y,p),[0 0.1 dtObs],XM);
    XM = sol(3,:)';
end
%%
save('DATAc.mat','zSpread','fSpread','locRad','dtObs','dxObs','p','obsErr',...
    'Xt','FM','AM');
