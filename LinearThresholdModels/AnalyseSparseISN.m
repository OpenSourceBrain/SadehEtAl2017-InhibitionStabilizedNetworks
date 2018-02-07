function [sResults] = AnalyseSparseISN(nNumNeurons, fPropInh, fTotExc, fTotInh, fEConnFillFactor, fIConnFillFactor, fPropRand)

vfBoostAmp = linspace(-1, 1, 21);
vfPropInhStim = linspace(0, 1, 31);

[mfBoostAmp, mfPropInhStim] = meshgrid(vfBoostAmp, vfPropInhStim);

if (~isstruct(nNumNeurons))
   
   % - Get neuron properties
   sNetwork = BuildSimpleNetwork(0.001);
   
   % - Construct a network weight matrix
   nNumSSN = 1;
   fSSN = 0.2;
   nNumExc = round(nNumNeurons*(1-fPropInh));
   nNumInh = nNumNeurons - nNumExc;
   
   [sNetwork.mfWeights, sMetrics] = ParameterisedSubnetwork(nNumNeurons, fPropInh, nNumSSN, ...
      fSSN, inf, 2, fTotExc, fTotInh, fEConnFillFactor, fIConnFillFactor);
   
   sNetwork.vnClasses = [ones(nNumExc, 1); 2*ones(nNumInh, 1)];
   sNetwork.vsNeurons = [repmat(sNetwork.vsNeurons(1), nNumExc, 1); repmat(sNetwork.vsNeurons(end), nNumInh, 1)];
   
   %% Numerically estimate proportion of exc. neurons that show paradoxical change
   % in inhbitory input current, for network defined at top
   
   cvfInhInputBase = cell(size(mfBoostAmp));
   cvfInhInputBoost = cell(size(mfBoostAmp));
   mfStimDeriv = nan(size(mfBoostAmp));
   mfNonStimDeriv = nan(size(mfBoostAmp));
   mfPropPara = nan(size(mfBoostAmp));
   mfExcChange = nan(size(mfBoostAmp));
   mfInhChange = nan(size(mfBoostAmp));
   parfor (nParamIndex = 1:numel(mfBoostAmp))
      fprintf('Parameter [%d/%d]\n', nParamIndex, numel(mfBoostAmp));
      nNumStim = round(mfPropInhStim(nParamIndex) .* nNumInh);
      [mfStimDeriv(nParamIndex), mfNonStimDeriv(nParamIndex), ...
         mfPropPara(nParamIndex), mfExcChange(nParamIndex), mfInhChange(nParamIndex), ...
         cvfInhInputBase{nParamIndex}, cvfInhInputBoost{nParamIndex}] = ...
         CalculatePropParadoxical(nNumStim, nNumInh, sNetwork, mfBoostAmp(nParamIndex), fPropRand, false);
   end
   
   sResults.sMetrics = sMetrics;
   sResults.nNumNeurons = nNumNeurons;
   sResults.fPropInh = fPropInh;
   sResults.fTotExc = fTotExc;
   sResults.fTotInh = fTotInh;
   sResults.fEConnFillFactor = fEConnFillFactor;
   sResults.fIConnFillFactor = fIConnFillFactor;
   sResults.mfStimDeriv = mfStimDeriv;
   sResults.mfNonStimDeriv = mfNonStimDeriv;
   sResults.mfPropPara = mfPropPara;
   sResults.mfExcChange = mfExcChange;
   sResults.mfInhChange = mfInhChange;
   sResults.vfBoostAmp = vfBoostAmp;
   sResults.vfPropInhStim = vfPropInhStim;
   sResults.fPropRand = fPropRand;
   sResults.cvfInhInputBase = cvfInhInputBase;
   sResults.cvfInhInputBoost = cvfInhInputBoost;
   
else
   sResults = nNumNeurons;
   sMetrics = sResults.sMetrics;
   nNumNeurons = sResults.nNumNeurons;
   fPropInh = sResults.fPropInh;
   fTotExc = sResults.fTotExc;
   fTotInh = sResults.fTotInh;
   fEConnFillFactor = sResults.fEConnFillFactor;
   fIConnFillFactor = sResults.fIConnFillFactor;
   mfPropPara = sResults.mfPropPara;
   mfExcChange = sResults.mfExcChange;
   mfInhChange = sResults.mfInhChange;
   vfBoostAmp = sResults.vfBoostAmp;
   vfPropInhStim = sResults.vfPropInhStim;
   fPropRand = sResults.fPropRand;
   cvfInhInputBase = sResults.cvfInhInputBase;
   cvfInhInputBoost = sResults.cvfInhInputBoost;
end


% -- Power tests, single-tailed
mnSampSizeReqd = nan(size(mfBoostAmp));
mfEstPower = nan(size(mfBoostAmp));
parfor (nParamIndex = 1:numel(mfBoostAmp))
   fMeanDiff = sign(mfBoostAmp(nParamIndex)) * (mean(cvfInhInputBoost{nParamIndex} - cvfInhInputBase{nParamIndex}));
   fStdDiff = std(cvfInhInputBoost{nParamIndex} - cvfInhInputBase{nParamIndex});
   
   if (fMeanDiff <= 0)
      mnSampSizeReqd(nParamIndex) = inf;
      mfEstPower(nParamIndex) = 0;
   else
      % -- Power test (beta = 0.9)
      mnSampSizeReqd(nParamIndex) = sampsizepwr('t', [fMeanDiff fStdDiff], 0, 0.9, [], 'Tail', 'left');

      % -- Estimated power (N = 100)
      mfEstPower(nParamIndex) = sampsizepwr('t', [fMeanDiff fStdDiff], 0, [], 100, 'Tail', 'left');
   end
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
h1 = subplot(1, 2, 1);
imagesc(vfBoostAmp, vfPropInhStim * 100, mnSampSizeReqd);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'wo', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 3, 'FontSize', 24);
xlabel('Boost amplitude \delta');
ylabel('Prop. inh stim. p/N_I (%)');
title('Samp. size for Pow=0.9');
axis xy;
colormap parula;
hCBar = colorbar;
caxis([0 1000]);
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Sample size for Pow=0.9');

h2 = subplot(1, 2, 2);
imagesc(vfBoostAmp, vfPropInhStim * 100, mfEstPower * 100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'wo', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 3, 'FontSize', 24);
xlabel('Boost amplitude \delta');
ylabel('Prop. inh stim. p/N_I (%)');
title('Est. pow. for N=100');
axis xy;
colormap parula;
caxis([0 100]);
hCBar = colorbar;
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Estimated power for N=100');


figure;
h1 = subplot(1, 3, 1);
imagesc(vfBoostAmp, vfPropInhStim*100, mfPropPara*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'wo', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 3, 'FontSize', 24);
xlabel('Boost amplitude \delta');
ylabel('Prop. inh stim. p/N_I (%)');
title('Prop. paradoxical (%)');
axis xy;
colormap parula;
caxis([0 100]);
hCBar = colorbar;
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Prop. paradoxical (%)');

h2=subplot(1, 3, 2);
imagesc(vfBoostAmp, vfPropInhStim*100, mfExcChange*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'ko', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 3, 'FontSize', 24);
xlabel('Boost amplitude \delta');
% ylabel('Prop. inh stim. p/N_I (%)');
title('Prop. exc. change (%)');
axis xy;
colormap parula;
caxis([-100 200]);
hCBar = colorbar;
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Prop. exc. change (%)');

h3=subplot(1, 3, 3);
imagesc(vfBoostAmp, vfPropInhStim*100, mfInhChange*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'ko', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 3, 'FontSize', 24);
xlabel('Boost amplitude \delta');
% ylabel('Prop. inh stim. p/N_I (%)');
title('Prop. stim. inh. change (%)');
axis xy;
caxis([-100 200]);
hCBar = colorbar;
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Prop. inh. stim. change (%)');

set(gcf, 'Position', [0 0 1500 400]);
set(h1, 'Position', [0.1    0.1575    0.15    0.7350]);
set(h2, 'Position', [0.41    0.1575    0.15    0.7350]);
set(h3, 'Position', [0.71    0.1575    0.15    0.7350]);

strFigName = sprintf('sparse-isn-prop-para-N%d-we%.2f-wi%.2f-he%.4f-hi%.4f', ...
   nNumNeurons, fTotExc, fTotInh, fEConnFillFactor(1), fIConnFillFactor(1));
hgsave(gcf, [strFigName '.fig']);
export_to_pdf(gcf, [strFigName '.pdf'], [], get(gcf, 'Position'));
