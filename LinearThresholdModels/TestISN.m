
% - Get neuron properties
sNetwork = BuildSimpleNetwork(0.001);

% - Construct a network weight matrix
nNumNeurons = 5000;
fPropInh = 0.2;
fSpecScale = 2.9;
fEConnFillFactor = fSpecScale .* [.0022 .072];
fIConnFillFactor = fSpecScale .* [.084 .34];
nNumSSN = 1;
fSSN = 0.2;
fTotExc = 1;5.4;
fTotInh = 1000;56;
nNumExc = round(nNumNeurons*(1-fPropInh));
nNumInh = nNumNeurons - nNumExc;

[sNetwork.mfWeights, sMetrics] = ParameterisedSubnetwork(nNumNeurons, fPropInh, nNumSSN, ...
   fSSN, inf, 2, fTotExc, fTotInh, fEConnFillFactor, fIConnFillFactor);
clear vfEigVals;

sNetwork.vnClasses = [ones(nNumExc, 1); 2*ones(nNumInh, 1)];
sNetwork.vsNeurons = [repmat(sNetwork.vsNeurons(1), nNumExc, 1); repmat(sNetwork.vsNeurons(end), nNumInh, 1)];


%% - Analyse eigenvalues

if ~exist('vfEigVals', 'var')
   [mfEigVecs, vfEigVals] = eigs(sNetwork.mfWeights, 50, 'lm');
   vfEigVals = complex(diag(vfEigVals));
   
   % - Find trivial eigenmode
   [~, nTrivialEig] = nanmin(abs(vfEigVals - sMetrics.fPredTrivialEigenvalue));
end

figure;
plot(vfEigVals, 'kx');
hold all;
vfTheta = linspace(0, 2*pi, 100);
plot(cos(vfTheta), sin(vfTheta), 'r-', 'LineWidth', 2);
plot(sMetrics.fExpectedRadius .* cos(vfTheta), sMetrics.fExpectedRadius .* sin(vfTheta), 'k-');
set(gca, 'LineWidth', 2, 'FontSize', 48);
xlabel('Re(\lambda)');
ylabel('Im(\lambda)');
axis equal;
xlim(5*[-1 1]);


%% - Analyse fraction of excitatory network needed for ISN property


vfEStrength = [0.5 1 2 5 10 20];

clear mfExcEigFracActive;
   
for (nEStrength = 1:numel(vfEStrength))

   % - Build a nominal network
   mfNominalWeights = ParameterisedSubnetwork(100, 0, 1, 0, inf, 2, vfEStrength(nEStrength), 0);
   
   for (nNumActive = 1:100)
      mfThisNet = mfNominalWeights(1:nNumActive, 1:nNumActive);
      mfExcEigFracActive(nEStrength, nNumActive) = real(eigs(mfThisNet, 1, 'la'));
   end
end

figure;
plot(mfExcEigFracActive);
legend(vfEStrength);

%% Analytical solution for Exc. ISN property over proportion of inactivation

vfEStrength = linspace(0, 10, 500);
vfFracActive = linspace(0, 1, 500);
fPropInh = 0.2;

[mfEStrength, mfFracActive] = ndgrid(vfEStrength, vfFracActive);
mfParams = [mfEStrength(:) mfFracActive(:)];

clear vfEigPropAct;
for (nParam = 1:size(mfParams, 1))
   fEStrength = mfParams(nParam, 1) * (1-fPropInh);
   fFracActive = mfParams(nParam, 2);
   vfEigPropAct(nParam) = fEStrength * fFracActive;
end

mfEigPropAct = reshape(vfEigPropAct, numel(vfEStrength), numel(vfFracActive))';

fhMinActiveForISN = @(we, fI)min(1, 1./(we*(1-fI)));

figure;
imagesc(vfEStrength, vfFracActive*100, mfEigPropAct);
hold all;
contour(vfEStrength, vfFracActive*100, mfEigPropAct, [1 1], 'k--', 'LineWidth', 1);
axis xy;
plot(5.4, fhMinActiveForISN(5.4, 0.2)*100, 'kx', 'MarkerSize', 8, 'LineWidth', 1);
set(gca, 'LineWidth', 1, 'FontSize', 6);
hC = colorbar;
colormap(flipud(gray(256)));
set(hC, 'LineWidth', 1, 'FontSize', 6);
ylabel(hC, 'Largest real eig. \lambda_E', 'FontSize', 8);
xlabel('Exc. strength w_E', 'FontSize', 8);
ylabel('Prop. exc. active (%)', 'FontSize', 8);
% title('Sufficient recurrent Exc. for ISN behaviour');
set(gca, 'Units', 'centimeters', 'OuterPosition', [0 0 8.5 5]);

%%
figure;
plot(vfEStrength, fhMinActiveForISN(vfEStrength, 0.2));

%% Simulate network

fPropInhStim = .16666;
tStimDelay = 120e-3;
fBoostAmp = -.3;
fStimAmp = 1;
fPropRand = 1;
tTimeStep = 5e-4;

nNumStim = ceil(nNumInh * fPropInhStim);
vfInputBase = fStimAmp * ((1-fPropRand) .* ones(nNumNeurons, 1) + fPropRand .* rand(nNumNeurons, 1));
vfInputBoost = fBoostAmp * fStimAmp * [zeros(nNumNeurons-nNumStim, 1); ones(nNumStim, 1)];

fhInputFunction = @(sN, tT, tDelta)vfInputBase.* (tT > tStimDelay/2) + vfInputBoost .* (tT > 1.5*tStimDelay);
fhMonitorFunction = @(varargin)MonitorTrace(varargin{:}, [1 nNumNeurons]);

[~, ~, ~, ~, mfResponseTrace, mfStateTrace] = SimulateNetwork(sNetwork, [], tTimeStep, tStimDelay*3, fhInputFunction, false, [1:nNumNeurons]);

% - Estimate inhibitory input over time
mfInhInput = sNetwork.mfWeights(:, nNumExc+1:end) * max(mfStateTrace(nNumExc+1:end, :), 0);

nBoostOnset = floor(tStimDelay*1.5 / tTimeStep)-1;

vfInhSS = mfInhInput(:, nBoostOnset);
vfInhBoost = mfInhInput(:, end);

vfBaseResp = mfResponseTrace(:, nBoostOnset);
vfBoostResp = mfResponseTrace(:, end);

% figure;
% plot(0, vfInhBoost - vfInhSS, 'kx');
% figure;
% ComparisonHistN((vfInhBoost - vfInhSS) ./ vfInhSS, 50);

fprintf('Proportion that change <10%%: %.2f\n', 100*nnz(abs((vfInhBoost - vfInhSS) ./ vfInhSS) < 0.1) ./ numel(vfInhBoost));

fprintf('Proportion that show paradoxical effect >10%%: %.2f\n', 100*nnz(-sign(fBoostAmp) .* ((vfInhBoost - vfInhSS) ./ vfInhSS) > 0.1) ./ numel(vfInhBoost));

vtTimeTrace = ((1:size(mfResponseTrace, 2))-1) .* tTimeStep ./ 1e-3;

figure;
plot(vtTimeTrace, mean(mfResponseTrace(1:nNumExc, :), 1)', 'r-', 'LineWidth', 5);
hold all;
plot(vtTimeTrace, mean(mfResponseTrace(end-nNumStim:end, :), 1)', 'b-', 'LineWidth', 5);
plot(vtTimeTrace, mean(mfResponseTrace(nNumExc+1:end-nNumStim-1, :), 1)', 'g--', 'LineWidth', 5);
set(gca, 'FontSize', 48, 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Activity (a.u.)');
% plot(mean(mfInhInput, 1)', ':');
legend('Exc.', 'Inh. (stim)', 'Inh.', 'I total input', 'Location', 'NorthWest');
title(sprintf('W_E=%.2f; W_I=%.2f; f_I=%.2f; %.0f%% inh stim; %.2f boost', fTotExc, fTotInh, fPropInh, fPropInhStim*100, fBoostAmp));
box off;

% export_to_pdf(gcf, 'isn-partial-stim-reduce-inh.pdf');

figure;
plot(1000*tTimeStep*((1:size(mfResponseTrace, 2))-1), mfResponseTrace', 'LineWidth', 5);
legend('Exc.', 'Inh.', 'Location', 'NorthWest');
set(gca, 'FontSize', 48, 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Activity (a.u.)');
box off;

figure;
plot(1000*tTimeStep*((1:size(mfResponseTrace, 2))-1), mfInhInput', 'b-', 'LineWidth', 5);
set(gca, 'FontSize', 48, 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Inhibitory input current to excitatory neurons (a.u.)');
box off;


%% Numerically calculate fixed-point derivative for network defined at top

fStimAmp = 1;
fBoostAmp = 0.01;

vnNumStim = unique(ceil(linspace(1, nNumInh, 10)));
vfPropInh = vnNumStim / nNumInh * 100;

vbActivateTrivial = false(numel(vfPropInh), 1);
clear vfInhDiff vfExcDiff vbInvalid
parfor (nStimIndex = 1:numel(vnNumStim))
   %%
   nNumStim = vnNumStim(nStimIndex);
   
   vfInputBase = [ones(400,1); zeros(400, 1); ones(200, 1)];%vfInputBase = fStimAmp * ones(nNumNeurons, 1);
   vfInputBoost = fBoostAmp * fStimAmp * [zeros(nNumNeurons-nNumStim, 1); ones(nNumStim, 1)];
   
   fhFPResp = @(t, y, vfI)sum((sNetwork.mfWeights * y - y + vfI).^2)./nNumNeurons; %#ok<PFBNS>
   
   vfBaseResp = fminunc(@(x)fhFPResp(0, x, vfInputBase), rand(nNumNeurons, 1));
   vfBoostResp = fminunc(@(x)fhFPResp(0, x, vfInputBase + vfInputBoost), vfBaseResp-vfInputBoost .* rand(nNumNeurons, 1));
   vfStimDiff = vfBoostResp - vfBaseResp;

   fhInputFunction = @(sN, tT, tDelta)vfInputBase + vfInputBoost .* (tT > tStimDelay);
   fhMonitorFunction = @(varargin)MonitorTrace(varargin{:}, [1 nNumNeurons]);

%    [~, ~, ~, ~, ~, mfStateTrace] = ...
%       SimulateNetwork(sNetwork, [], tTimeStep, tStimDelay*2, fhInputFunction, false, [1 nNumNeurons]);
%    nStimSteps = size(mfStateTrace, 2);
%    vfBaseResp = mfStateTrace(:, floor(nStimSteps/2));
%    vfBoostResp = mfStateTrace(:, nStimSteps);
%    vfStimDiff = vfBoostResp - vfBaseResp;
%    
   vbInvalid(nStimIndex) = any(vfBoostResp <= 0);
   
   vfInhDiff(nStimIndex) = nanmean(vfStimDiff(nNumNeurons-nNumStim+1:end));
   vfExcDiff(nStimIndex) = nanmean(vfStimDiff(1:nNumExc));
   
   fprintf('%d/%d\n', nStimIndex, numel(vnNumStim));
end

figure;
plot(vfPropInh, vfInhDiff);
hold all;
plot(xlim, [0 0], 'k:');
plot(vfPropInh(vbInvalid), vfInhDiff(vbInvalid), 'rx');


%% Numerically estimate proportion of inhibitory network required to perturb 
%  in an homogenous network with f_I% inhibitory neurons

vfExcWeight = linspace(0, 10, 19);
vfInhWeight = linspace(0, 100, 29);

[mfExc, mfInh] = meshgrid(vfExcWeight, vfInhWeight);

% - Fixed network params
nNumNeurons = 1000;
fPropInh = 0.2;
fEConnFillFactor = 1;[0.2 0.5];0.01;
fIConnFillFactor = 1;[0.5 0.5];0.01;
nNumSSN = 2;
fSSN = 0.1;

mfP = nan(size(mfExc));
mfDeriv = nan(size(mfExc));
mnOptFlag = nan(size(mfExc));
mfEigGlobal = nan(size(mfExc));
mfEigSSN = nan(size(mfExc));
mfEigNonSSN = nan(size(mfExc));

parfor (nParamIndex = 1:numel(mfExc))
   fprintf('Param [%d/%d]\n', nParamIndex, numel(mfExc));
   
   % - Construct a network weight matrix
   fTotExc = mfExc(nParamIndex);
   fTotInh = mfInh(nParamIndex);
   nNumExc = round(nNumNeurons*(1-fPropInh));
   nNumInh = nNumNeurons - nNumExc;
   
   %% - Test for ISN property
   bNonISN = fTotExc <= 1;
   if (bNonISN)
      continue;
   end
   
   %% - Build a full network
   sThisNet = BuildSimpleNetwork(0.001);
   [sThisNet.vsClasses.cActNoiseParams] = deal({0 0});

   [sThisNet.mfWeights, sMetrics] = ParameterisedSubnetwork(nNumNeurons, fPropInh, nNumSSN, ...
      fSSN,inf, 4, fTotExc, fTotInh, fEConnFillFactor, fIConnFillFactor);
   
   sThisNet.vnClasses = [ones(nNumExc, 1); 2*ones(nNumInh, 1)];
   sThisNet.vsNeurons = [repmat(sThisNet.vsNeurons(1), nNumExc, 1); repmat(sThisNet.vsNeurons(end), nNumInh, 1)];
   
   %% - Test global network stability (without SSNs)
   
   [mfWeightsNonSSN] = ParameterisedSubnetwork(nNumNeurons, fPropInh, 1, ...
      fSSN, inf, 1, fTotExc, fTotInh, fEConnFillFactor, fIConnFillFactor);   
   
   vfMaxEigNonSSN = 0;
   vfMaxEigSSN = 0;
   vfMaxEigGlobal = 0;
   try
      vfMaxEigNonSSN = eigs(mfWeightsNonSSN, 1, 'lr');
      
      if (nNumSSN > 1) && (fSSN > 0)
         vfMaxEigGlobal = eigs(sThisNet.mfWeights, 1, 'lr');
         mfWeightsSSN = sThisNet.mfWeights([1:400 801:end], [1:400 801:end]);         
         vfMaxEigSSN = eigs(mfWeightsSSN, 1, 'lr');
      end
      
   catch
      try
         vfMaxEigNonSSN = eig(mfWeightsNonSSN);
         if (nNumSSN > 1) && (fSSN > 0)
            vfMaxEigGlobal = eig(sThisNet.mfWeights);
            mfWeightsSSN = sThisNet.mfWeights([1:400 801:end], [1:400 801:end]);
            vfMaxEigSSN = eig(mfWeightsSSN);
         end
      catch
         continue;
      end
   end
   
   mfEigGlobal(nParamIndex) = max(real(vfMaxEigGlobal));
   mfEigNonSSN(nParamIndex) = max(real(vfMaxEigNonSSN));
   mfEigSSN(nParamIndex) = max(real(vfMaxEigSSN));   
   
   if (max(real(vfMaxEigNonSSN)) >= 1) || (max(real(vfMaxEigSSN)) >= 1)
      % - Unstable network
      continue;
   end
   
   %% - Estimate proportion for this parameter combination
%    tic;
   [mfP(nParamIndex), mfDeriv(nParamIndex), mnOptFlag(nParamIndex)] = ...
      fzero_LINEAR(@(p)(NumericPerturbationDeriv(p*nNumInh, nNumInh, sThisNet, false)), 0.5);
%    toc;
   
end

clear nNumInh;


figure;
imagesc(vfExcWeight, vfInhWeight, mfP * 100);
hold all;
plot(sNetwork.vsNeurons(1).fSynapseWeight .* sNetwork.vsNeurons(1).nAxonalNumSynapses, ...
     -sNetwork.vsNeurons(end).fSynapseWeight .* sNetwork.vsNeurons(end).nAxonalNumSynapses, 'kx', 'MarkerSize', 48, 'LineWidth', 3);
axis xy tight;
hC = colorbar;
set(gca, 'FontSize', 48, 'LineWidth', 2, 'XTick', [1 5 10]);
set(hC, 'FontSize', 48, 'LineWidth', 2);
xlabel('w_E');
ylabel('w_I');
ylabel(hC, 'p/N_I (%)');
caxis([0 100]);
colormap(parula(512));


%% Numerically estimate proportion of exc. neurons that show paradoxical change
% in inhbitory input current, for network defined at top

fPropRand = 1;
vfBoostAmp = linspace(-1, 1, 29);
vfPropInhStim = linspace(0, 1, 39);

[mfBoostAmp, mfPropInhStim] = meshgrid(vfBoostAmp, vfPropInhStim);

mfPropPara = nan(size(mfBoostAmp));
mfExcChange = nan(size(mfBoostAmp));
mfInhChange = nan(size(mfBoostAmp));
parfor (nParamIndex = 1:numel(mfBoostAmp))
   fprintf('Parameter [%d/%d]\n', nParamIndex, numel(mfBoostAmp));
   nNumStim = round(mfPropInhStim(nParamIndex) .* nNumInh);
   [~, ~, mfPropPara(nParamIndex), mfExcChange(nParamIndex), mfInhChange(nParamIndex)] = ...
      CalculatePropParadoxical(nNumStim, nNumInh, sNetwork, mfBoostAmp(nParamIndex), fPropRand, false);
end


fTargetEResp = [1.4 -.3];
fTargetIResp = [-.4 1.2];
fTolerance = 0.2;

mbTarget = false(size(mfBoostAmp));

for (nTarget = 1:numel(fTargetEResp))
   mbTarget = mbTarget | ...
              (mfExcChange > (fTargetEResp(nTarget)-fTolerance)) & (mfExcChange < (fTargetEResp(nTarget)+fTolerance)) & ...
              (mfInhChange > (fTargetIResp(nTarget)-fTolerance)) & (mfInhChange < (fTargetIResp(nTarget)+fTolerance));
end

figure;
subplot(1, 3, 1);
imagesc(vfBoostAmp, vfPropInhStim*100, mfPropPara*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'ko', 'MarkerSize', 12);
xlabel('Relative boost amplitude (a.u.)');
ylabel('Prop. inh stim. p/N_I (%)');
title(sprintf('Random fraction: %.2f%%', fPropRand*100));
axis xy;
colormap parula;
hCBar = colorbar;
ylabel(hCBar, 'Prop. paradoxical (%)');

subplot(1, 3, 2);
imagesc(vfBoostAmp, vfPropInhStim*100, mfExcChange*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'ko', 'MarkerSize', 12);
xlabel('Relative boost amplitude (a.u.)');
ylabel('Prop. inh stim. p/N_I (%)');
title(sprintf('Random fraction: %.2f%%', fPropRand*100));
axis xy;
colormap parula;
caxis([-100 200]);
hCBar = colorbar;
ylabel(hCBar, 'Prop. exc. change (%)');

subplot(1, 3, 3);
imagesc(vfBoostAmp, vfPropInhStim*100, mfInhChange*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'ko', 'MarkerSize', 12);
xlabel('Relative boost amplitude (a.u.)');
ylabel('Prop. inh stim. p/N_I (%)');
title(sprintf('Random fraction: %.2f%%', fPropRand*100));
axis xy;
caxis([-100 200]);
hCBar = colorbar;
ylabel(hCBar, 'Prop. inh. stim. change (%)');
set(gcf, 'Position', [0 0 1200 300]);

%% Examine paradoxical proportion over a range of parameters

vfTotE = 4;

clear vsResults;
for (nParamIndex = 1:numel(vfTotE))
   [vsResults(nParamIndex)] = AnalyseSparseISN(5000, 0.2, vfTotE(nParamIndex), 100, fEConnFillFactor, fIConnFillFactor, 1.0);
%    AnalyseSparseISN(vsResults(nParamIndex));
end

save sparse-sim-large_4_100.mat vsResults;

%%

nResult = 1;
nPropInd = 5;

figure;
hold all;
for (nBoostInd = 1:numel(vsResults(nResult).vfBoostAmp))
   plot(vsResults(nResult).cvfInhInputBoost{nPropInd, nBoostInd}, vsResults(nResult).cvfInhInputBase{nPropInd, nBoostInd}, '.')
%    plot((vsResults(nResult).cvfInhInputBoost{nPropInd, nBoostInd} - vsResults(nResult).cvfInhInputBase{nPropInd, nBoostInd}) ./ vsResults(nResult).cvfInhInputBase{nPropInd, nBoostInd}, 'k.');
end

plot([0 -2], [0 -2], 'k:');
axis equal;

%% Resample results to perform Monte-Carlo power analysis

nNumBootstraps = 200;
nMaxN = 200;

s = vsResults(1);

cmfPower = cell(size(s.mfPropPara));

[mfBoostAmp, mfPropInhStim] = meshgrid(s.vfBoostAmp, s.vfPropInhStim);

parfor (nParam = 1:numel(s.mfPropPara))
   fprintf('Param [%d/%d]\n', nParam, numel(s.mfPropPara));
   vfPower = [];

   % - Loop over sample sizes
   for nN = 1:nMaxN
      vfBootDiff = [];
      for (nBoot = 1:nNumBootstraps)
         % - Skip empty results
         if (isnan(s.cvfInhInputBase{nParam}))
            continue;
         end
         
         % - Choose N neurons
         vnSample = randsample(numel(s.cvfInhInputBase{nParam}), nN, true);
         
         % - Compute average delta, taking only cells that change their
         % responses by more than 10%
         vfDiffProp = -sign(mfBoostAmp(nParam)) .* (s.cvfInhInputBoost{nParam}(vnSample) - s.cvfInhInputBase{nParam}(vnSample)) ./ s.cvfInhInputBase{nParam}(vnSample);
         vfBootDiff(nBoot) = mean(vfDiffProp(abs(vfDiffProp) > 0.1));
      end
      
      % - Estimate power for this sample size
      vfPower(nN) = nnz(vfBootDiff > 0.1) ./ nNumBootstraps;
   end
   
   cmfPower{nParam} = vfPower;
end


   
%% Make a figure showing the range for which w_E can show paradoxical effect in a single neuron, for homogenous equal E/I size networks

wi=1;
vnN = linspace(1, 100, 1000);

figure;
plot(vnN, 1+wi, 'r-', 'LineWidth', 3);
hold all;
plot(vnN, 1+(vnN-1).*wi./vnN, 'r-', 'LineWidth', 3);
set(gca, 'FontSize', 48, 'LineWidth', 2);
box off;
xlabel('Network size N');
ylabel('w_E');
ylim([.9 2.1]*wi);
xlim([0 max(vnN)]);
set(gca, 'XTick', [1 50 100]);

% export_to_pdf(gcf, 'single-inh-stim-we-constraint.pdf');

%% Make a figure showing proportion needed to stimulate to see paradoxical effect in homogenous network with equal E/I size

vfExcWeight = linspace(0, 10, 500);
vfInhWeight = linspace(0, 20, 2000);

[mfExc, mfInh] = meshgrid(vfExcWeight, vfInhWeight);

mbUnstable = mfExc >= (1+mfInh);
mbNonISN = mfExc <= 1;

mfP = (1-mfExc+mfInh)./(mfInh);
mfP(mfP<0) = nan;
mfP(mfP>1) = nan;
mfP(mbUnstable) = nan;
mfP(mbNonISN) = nan;
figure;
imagesc(vfExcWeight, vfInhWeight, mfP);
axis xy equal tight;
hC = colorbar;
set(gca, 'FontSize', 48, 'LineWidth', 2, 'XTick', [1 5 10]);
set(hC, 'FontSize', 48, 'LineWidth', 2);
xlabel('w_E');
ylabel('w_I');
ylabel(hC, 'p/N');
caxis([0 1]);
colormap(parula(512));

% export_to_ai(gcf, 'prop-stim-paradox-we-we.ai');
% axis off; colorbar off;
% export_to_png(gcf, 'prop-stim-paradox-we-we.png');

%% Make an image to use for a colorbar
figure;
imagesc((512:-1:1)');
colormap(parula(512));
axis off;
% export_to_png(gcf, 'parula-colorbar.png');

%% Runs some simulations of the physical model to explore parameter space

vfEAxonWidths = linspace(100e-6, 1200e-6, 19);
vfIAxonWidths = linspace(100e-6, 600e-6, 13);

[mfE, mfI] = ndgrid(vfEAxonWidths, vfIAxonWidths);
mfAWParams = [mfE(:) mfI(:)];

clear cvfPertResp cvfPertWidths;
parfor (nParam = 1:numel(mfE))
   vfAxonWidths = mfAWParams(nParam, :);
   
   [cvfPertRespAxon{nParam}, cvfPertWidthsAxon{nParam}] = PhysicalSimulation(vfAxonWidths, [], [], [], [], false);
   fprintf('----- Parameter combination [%d/%d]\n', nParam, numel(mfE));
end


%% Runs some simulations of the physical model to explore parameter space

vfEDendWidths = linspace(50e-6, 500e-6, 17);
vfIDendWidths = linspace(50e-6, 500e-6, 17);

[mfE, mfI] = ndgrid(vfEDendWidths, vfIDendWidths);
mfDWParams = [mfE(:) mfI(:)];

clear cvfPertRespDend cvfPertWidthsDend;
parfor (nParam = 1:numel(mfE))
   vfDendWidths = mfDWParams(nParam, :);
   
   [cvfPertRespDend{nParam}, cvfPertWidthsDend{nParam}] = PhysicalSimulation([], vfDendWidths, [], [], [], false);
   fprintf('----- Parameter combinatin [%d/%d]\n', nParam, numel(mfE));
end


%% Runs some simulations of the physical model to explore parameter space

vfEWeight = linspace(1, 10, 17);
vfIWeight = linspace(1, 10, 17);

[mfE, mfI] = ndgrid(vfEWeight, vfIWeight);
mfWParams = [mfE(:) mfI(:)];

clear cvfPertRespWeight cvfPertWidthsWeight;
parfor (nParam = 1:numel(mfE))
   vfWeights = mfWParams(nParam, :);
   
   [cvfPertRespWeight{nParam}, cvfPertWidthsWeight{nParam}] = PhysicalSimulation([], [], vfWeights(1), vfWeights(2), [], false);
   fprintf('----- Parameter combination [%d/%d]\n', nParam, numel(mfE));
end


%% Locate paradoxical threshold

PlotPerturbWidthThreshold(cvfPertWidthsDend, cvfPertRespDend, vfEDendWidths*1e6, vfIDendWidths*1e6, ...
   'E dend. width ({\mu}m)', 'I dend. width ({\mu}m)');
caxis([0 250]);
set(gca, 'Units', 'centimeters', 'Position', [2 2 8 3]);


PlotPerturbWidthThreshold(cvfPertWidthsAxon, cvfPertRespAxon, vfEAxonWidths*1e6, vfIAxonWidths*1e6, ...
   'E axon width ({\mu}m)', 'I axon width ({\mu}m)');
caxis([0 250]);
set(gca, 'Units', 'centimeters', 'Position', [2 2 8 3]);


PlotPerturbWidthThreshold(cvfPertWidthsWeight, cvfPertRespWeight, vfEWeight, vfIWeight, 'w_E', 'w_I');
caxis([0 250]);
set(gca, 'Units', 'centimeters', 'Position', [2 2 8 3]);



