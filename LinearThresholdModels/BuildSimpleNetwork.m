function [sNetwork, vsSN, fNeuronDensity2D, vfSpace] = BuildSimpleNetwork(fCortexScale, fExcFactor, fInhFactor)

%% Network parameters

cActNoiseParams = {0, .005};
fActNoiseWeight = 1;

if (~exist('fCortexScale', 'var') || isempty(fCortexScale))
   fCortexScale = 0.02;
end

if (~exist('fExcFactor', 'var'))
   fExcFactor = 1;
end

if (~exist('fInhFactor', 'var'))
   fInhFactor = 1;
end

fhInitialStateFunction = @InitialStateWhite;
cInitialStateParams = {0, 0};

fNeuronDensity3D = 146154;    % Neurons per cubic mm, layers 2&3 of the rodent (Schuez and Palm 1989)
fCorticalThickness = .2463 * fCortexScale;    % mm (Schuez and Palm 1989
fNeuronDensity2D = fNeuronDensity3D * fCorticalThickness / 1e-3 / 1e-3; % Neurons per square meter of our region of cortex

fExcWeightPerSynapse = 0.1 * 0.1 ./ fCortexScale;   %pC
fExcGain = 0.066;   % 1 / pC
nNumExcSynapses = 5696 * 2 * .7147 * fCortexScale;    % Binzegger 2004 (including estimate for rodent cortex)
fExcDendWidth = 300e-6;
fExcAxonWidth = sqrt((1200e-6)^2 - fExcDendWidth^2);  % Estimates from connectivity studies

fInhWeightPerSynapse = -0.1 * 0.1 * 10 ./ fCortexScale;
fInhGain = fExcGain;
nNumInhSynapses = 9678 * .8851 * fCortexScale;    % Binzegger 2004 (including estimate for rodent cortex)
fInhAxonWidth = 400e-6;
fInhDendWidth = 300e-6;

fInhProportion = 0.18;

vfSpace = fExcAxonWidth * [4 4];


%% Configure network

nNumNeurons = round(fNeuronDensity2D * prod(vfSpace));
sNetwork.fCortexScale = fCortexScale;

sBaseExcNeuron.nClass = uint8(1);
sBaseExcNeuron.vfLocation = [nan nan];
sBaseExcNeuron.fSynapseWeight = fExcFactor * fExcWeightPerSynapse * fExcGain;           % Assuming gain of 1
sBaseExcNeuron.fAxonalSigma = fExcAxonWidth / 4;
sBaseExcNeuron.nAxonalNumSynapses = nNumExcSynapses;
sBaseExcNeuron.fDendriticSigma = fExcDendWidth / 4;
sBaseExcNeuron.nDendriticNumSynapses = nan;
sBaseExcNeuron.nMaxSynapsesPerPartner = uint8(5);

sBaseInhNeuron.nClass = uint8(2);
sBaseInhNeuron.vfLocation = [nan nan];
sBaseInhNeuron.fSynapseWeight = fInhFactor * fInhWeightPerSynapse * fInhGain;           % Assuming gain of 1
sBaseInhNeuron.fAxonalSigma = fInhAxonWidth / 4;
sBaseInhNeuron.nAxonalNumSynapses = nNumInhSynapses;
sBaseInhNeuron.fDendriticSigma = fInhDendWidth / 4;
sBaseInhNeuron.nDendriticNumSynapses = nan;
sBaseInhNeuron.nMaxSynapsesPerPartner = uint8(5);


%% Build excitatory neurons

nNumExcNeurons = round(nNumNeurons * (1-fInhProportion));
[mfLoc] = HardBallProcess2D(nNumExcNeurons, vfSpace, 0);
% mfLocs = [linspace(0, vfSpace(1), round(sqrt(nNumExcNeurons)))' linspace(0, vfSpace(2), round(sqrt(nNumExcNeurons)))'];
% [mfLocX, mfLocY] = meshgrid(mfLocs(:, 1), mfLocs(:, 2));
% mfLoc = [mfLocX(:) mfLocY(:)];

nNumExcNeurons = size(mfLoc, 1);
vsEN = repmat(sBaseExcNeuron, nNumExcNeurons, 1);

parfor (nExcNID = 1:nNumExcNeurons)
   vsEN(nExcNID).vfLocation = mfLoc(nExcNID, :);
end

%% Build inhibitory neurons

nNumInhNeurons = round(nNumNeurons * (fInhProportion));
[mfLoc] = HardBallProcess2D(nNumInhNeurons, vfSpace, 0);
% mfLocs = [linspace(0, vfSpace(1), round(sqrt(nNumInhNeurons)))' linspace(0, vfSpace(2), round(sqrt(nNumInhNeurons)))'];
% [mfLocX, mfLocY] = meshgrid(mfLocs(:, 1), mfLocs(:, 2));
% mfLoc = [mfLocX(:) mfLocY(:)];

nNumInhNeurons = size(mfLoc, 1);
vsIN = repmat(sBaseInhNeuron, nNumInhNeurons, 1);

parfor (nInhNID = 1:nNumInhNeurons)
   vsIN(nInhNID).vfLocation = mfLoc(nInhNID, :);
end


%% Return neuron structures and assign IDs
vsSN = [vsEN; vsIN];

cnIDs = num2cell(1:numel(vsSN));
[vsSN.nNeuronID] = deal(cnIDs{:});


%% - Set up network structure
sNetwork.vsNeurons = vsSN;
sNetwork.vfSpace = vfSpace;
sNetwork.fNeuronDensity = fNeuronDensity2D;

vnClasses(1:nNumExcNeurons) = uint8(1);
vnClasses(nNumExcNeurons+1:nNumExcNeurons+nNumInhNeurons) = uint8(2);
sNetwork.vnClasses = vnClasses;

sNetwork.vfAxonalSigmas = [repmat(sBaseExcNeuron.fAxonalSigma, nNumExcNeurons, 1);
                           repmat(sBaseInhNeuron.fAxonalSigma, nNumInhNeurons, 1)];
sNetwork.vfDendriticSigmas = [repmat(sBaseExcNeuron.fDendriticSigma, nNumExcNeurons, 1);
                              repmat(sBaseInhNeuron.fDendriticSigma, nNumInhNeurons, 1)];

eap.fThreshold = 0;
eap.fGain = 1;
eap.fTau = 10e-3;
eap.fTauNoiseSigma = 0e-3;

sClass1.strDescription = 'Excitatory neurons';
sClass1.fhActivationFunction = @LinearThresholdEuler;
sClass1.sActivationParams = eap;
sClass1.fhActNoiseFunc = @norm_diffusive_noise;
sClass1.cActNoiseParams = cActNoiseParams;
sClass1.fActNoiseWeight = fActNoiseWeight;
sClass1.fhInitialStateFunction = fhInitialStateFunction;
sClass1.cInitialStateParams = cInitialStateParams;

iap.fThreshold = 0;
iap.fGain = 1;
iap.fTau = 10e-3;
iap.fTauNoiseSigma = 0e-3;

sClass2.strDescription = 'Inhibitory neurons';
sClass2.fhActivationFunction = @LinearThresholdEuler;
sClass2.sActivationParams = iap;
sClass2.fhActNoiseFunc = @norm_diffusive_noise;
sClass2.cActNoiseParams = cActNoiseParams;
sClass2.fActNoiseWeight = fActNoiseWeight;
sClass2.fhInitialStateFunction = fhInitialStateFunction;
sClass2.cInitialStateParams = cInitialStateParams;

sNetwork.vsClasses = [sClass1 sClass2];

sNetwork.cmhClassOverlapFunctions = repmat({@UnbiasedFieldOverlapTorus}, [2 2]);

sNetwork.mfClassConnectionProbs = [(1-fInhProportion) fInhProportion; (1-fInhProportion) fInhProportion];
