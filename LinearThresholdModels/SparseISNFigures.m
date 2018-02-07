function SparseISNFigures(s, cmfPower)

vfBoostAmp = s.vfBoostAmp;
vfPropInhStim = s.vfPropInhStim;
[mfBoostAmp, mfPropInhStim] = meshgrid(vfBoostAmp, vfPropInhStim);

fTargetEResp = [1.4 -.3];
fTargetIResp = [-.4 1.2];
fTolerance = 0.2;

mbTarget = false(size(mfBoostAmp));

for (nTarget = 1:numel(fTargetEResp))
   mbTarget = mbTarget | ...
      (s.mfExcChange > (fTargetEResp(nTarget)-fTolerance)) & (s.mfExcChange < (fTargetEResp(nTarget)+fTolerance)) & ...
      (s.mfInhChange > (fTargetIResp(nTarget)-fTolerance)) & (s.mfInhChange < (fTargetIResp(nTarget)+fTolerance));
end


% - Estimate power for N
mfP100 = cellfun(@(c)c(100), cmfPower);
mfP10 = cellfun(@(c)c(10), cmfPower);
mfP1 = cellfun(@(c)c(1), cmfPower);

% - Estimate N for power = 0.9
fTargetPower = 0.9;
[~, cnNPower] = cellfun(@(c)find(c >= fTargetPower, 1, 'first'), cmfPower, 'UniformOutput', false);

% - Power for N=100
figure;
imagesc(vfBoostAmp, vfPropInhStim * 100, mfP100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'wo', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 2, 'FontSize', 24);
xlabel('Boost amplitude \delta');
ylabel('Prop. inh stim. p/N_I (%)');
title('Est. power for N=100');
axis xy;
colormap parula;
hCBar = colorbar;
caxis([0 1]);
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Sample size for Pow=0.9');


% - N for power = 0.9
% figure;
% imagesc(vfBoostAmp, vfPropInhStim * 100, mfEstPower * 100);
% hold all;
% plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'wo', 'MarkerSize', 24, 'LineWidth', 3);
% set(gca, 'LineWidth', 2, 'FontSize', 24);
% xlabel('Boost amplitude \delta');
% ylabel('Prop. inh stim. p/N_I (%)');
% title('Est. pow. for N=100');
% axis xy;
% colormap parula;
% caxis([0 100]);
% hCBar = colorbar;
% set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Estimated power for N=100');


figure;
imagesc(vfBoostAmp, vfPropInhStim*100, s.mfPropPara*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'wo', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 2, 'FontSize', 24);
xlabel('Boost amplitude \delta');
ylabel('Prop. inh stim. p/N_I (%)');
title('Prop. paradoxical (%)');
axis xy;
colormap parula;
caxis([0 100]);
hCBar = colorbar;
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Prop. paradoxical (%)');

% - Make an orange/blue colormap
fHueBlue = 0.65;
fHueOrange = 0.08;
nBlueLength = 128;
nOrangeLength = 256;

mfColormap = vertcat(flipud(WhiteToHue(fHueBlue, nBlueLength)), [0 0 0], WhiteToHue(fHueOrange, nOrangeLength));

figure;
imagesc(vfBoostAmp, vfPropInhStim*100, s.mfExcChange*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'ko', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 2, 'FontSize', 24);
xlabel('Boost amplitude \delta');
% ylabel('Prop. inh stim. p/N_I (%)');
title('Prop. exc. change (%)');
axis xy;
colormap(mfColormap);
caxis([-100 200]);
hCBar = colorbar;
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Prop. exc. change (%)');

figure;
imagesc(vfBoostAmp, vfPropInhStim*100, s.mfInhChange*100);
hold all;
plot(mfBoostAmp(mbTarget), mfPropInhStim(mbTarget)*100, 'ko', 'MarkerSize', 24, 'LineWidth', 3);
set(gca, 'LineWidth', 2, 'FontSize', 24);
xlabel('Boost amplitude \delta');
% ylabel('Prop. inh stim. p/N_I (%)');
title('Prop. stim. inh. change (%)');
axis xy;
colormap(mfColormap);
caxis([-100 200]);
hCBar = colorbar;
set(hCBar, 'LineWidth', 2, 'FontSize', 24);
% ylabel(hCBar, 'Prop. inh. stim. change (%)');

