nNumPoints = 2000;
fMaxDist = 3000;

vfX = linspace(-fMaxDist, fMaxDist, nNumPoints);
vfY = linspace(-fMaxDist, fMaxDist, nNumPoints);
fBinArea = median(diff(vfX)) .* median(diff(vfY));

fExcDendWidth = 300;
fExcAxonWidth = sqrt((1200)^2 - fExcDendWidth^2);  % Estimates from connectivity studies

fEAxonSigma = fExcAxonWidth/4;
fIAxonSigma = 300/4;
fEDendriteSigma = fExcDendWidth/4;
fIDendriteSigma = 300/4;

[mfX, mfY] = meshgrid(vfX, vfY);
mfD = sqrt(mfX.^2 + mfY.^2);

mfEEProb = GaussianOverlap(fEAxonSigma, fEDendriteSigma, mfD);
mfEIProb = GaussianOverlap(fEAxonSigma, fEDendriteSigma, mfD);
mfIEProb = GaussianOverlap(fIAxonSigma, fEDendriteSigma, mfD);
mfIIProb = GaussianOverlap(fIAxonSigma, fIDendriteSigma, mfD);

mfEEProb = mfEEProb ./ sum(mfEEProb(:));
mfEIProb = mfEIProb ./ sum(mfEIProb(:));
mfIEProb = mfIEProb ./ sum(mfIEProb(:));
mfIIProb = mfIIProb ./ sum(mfIIProb(:));


%%

fPropInhNeurons = 0.2;
fPropEISyns = 0.45;         % If different from fPropInhNeurons, indicates class specificity
fPropIESyns = 1-fPropInhNeurons;

fNeuronDensity3D = 146154;    % Neurons per cubic mm, layers 2&3 of the rodent (Schuez and Palm 1989)
fCorticalThickness = .2463;    % mm (Schuez and Palm 1989)
fNeuronDensity2D = fNeuronDensity3D * fCorticalThickness / 1e-3 / 1e-3 * 1e-6 * 1e-6; % Neurons per µmeter of our region of cortex

nNumExcSynapses = 5696 * 2 * .7147;    % Binzegger 2004 (including estimate for rodent cortex)
nNumInhSynapses = 9678 * .8851;    % Binzegger 2004 (including estimate for rodent cortex)

nNumENeuronsPerBin = fBinArea .* fNeuronDensity2D .* (1-fPropInhNeurons);
nNumINeuronsPerBin = fBinArea .* fNeuronDensity2D .* fPropInhNeurons;

mfEEExpectedSyns = mfEEProb .* (1-fPropEISyns) .* nNumExcSynapses;
mfEIExpectedSyns = mfEIProb .* (fPropEISyns) .* nNumExcSynapses;

mfIEExpectedSyns = mfIEProb .* (fPropIESyns) .* nNumInhSynapses;
mfIIExpectedSyns = mfIIProb .* (1-fPropIESyns) .* nNumInhSynapses;

mfEEConnProb = mfEEExpectedSyns ./ nNumENeuronsPerBin;
mfEIConnProb = mfEIExpectedSyns ./ nNumINeuronsPerBin;
mfIEConnProb = mfIEExpectedSyns ./ nNumENeuronsPerBin;
mfIIConnProb = mfIIExpectedSyns ./ nNumINeuronsPerBin;

fprintf('------   Global sparsity   --------------------------\n');
fEESparsity = sum(min(mfEEExpectedSyns(:), 1)) ./ (nNumENeuronsPerBin .* numel(mfD))
fEISparsity = sum(min(mfEIExpectedSyns(:), 1)) ./ (nNumINeuronsPerBin .* numel(mfD))
fIESparsity = sum(min(mfIEExpectedSyns(:), 1)) ./ (nNumENeuronsPerBin .* numel(mfD))
fIISparsity = sum(min(mfIIExpectedSyns(:), 1)) ./ (nNumINeuronsPerBin .* numel(mfD))
fprintf('-----------------------------------------------------\n');


fSparsityDist = 750;
mbWindow = mfD <= fSparsityDist;

fprintf('------   Window sparsity [%d um radius, %d neurons]  ------------\n', fSparsityDist, round(fBinArea .* fNeuronDensity2D .* nnz(mbWindow)));
fEESparsity = sum(min(mfEEExpectedSyns(mbWindow), 1)) ./ (nNumENeuronsPerBin .* nnz(mbWindow))
fEISparsity = sum(min(mfEIExpectedSyns(mbWindow), 1)) ./ (nNumINeuronsPerBin .* nnz(mbWindow))
fIESparsity = sum(min(mfIEExpectedSyns(mbWindow), 1)) ./ (nNumENeuronsPerBin .* nnz(mbWindow))
fIISparsity = sum(min(mfIIExpectedSyns(mbWindow), 1)) ./ (nNumINeuronsPerBin .* nnz(mbWindow))
fprintf('-----------------------------------------------------\n');


figure;
subplot(2, 2, 1);
imagesc(vfX, vfY, mfEEConnProb);
axis equal tight;
colorbar;

subplot(2, 2, 2);
imagesc(vfX, vfY, mfIEConnProb);
axis equal tight;
caxis([0 1]);
colorbar;

subplot(2, 2, 3);
imagesc(vfX, vfY, mfEIConnProb);
axis equal tight;
colorbar;

subplot(2, 2, 4);
imagesc(vfX, vfY, mfIIConnProb);
axis equal tight;
caxis([0 1]);
colorbar;

colormap parula;

%% Bin connection probability by distance

vfDistBin = [-inf linspace(10, 1500, 100)];

clear vfEEConnProb vfEIConnProb vfIEConnProb vfIIConnProb;
for (nBin = 1:numel(vfDistBin)-1)
   mbThisBin = (mfD >= vfDistBin(nBin)) & (mfD < vfDistBin(nBin+1));
   vfEEConnProb(nBin) = nanmean(mfEEConnProb(mbThisBin));
   vfEIConnProb(nBin) = nanmean(mfEIConnProb(mbThisBin));
   vfIEConnProb(nBin) = nanmean(mfIEConnProb(mbThisBin));
   vfIIConnProb(nBin) = nanmean(mfIIConnProb(mbThisBin));
end
%%
figure;
plot(vfDistBin(2:end), min(vfEEConnProb, 1) * 100, 'r-', 'LineWidth', 5);
hold all;
plot(vfDistBin(2:end), min(vfEIConnProb, 1) * 100, 'r:', 'LineWidth', 5);
plot(vfDistBin(2:end), min(vfIEConnProb, 1) * 100, 'b:', 'LineWidth', 5);
plot(vfDistBin(2:end), min(vfIIConnProb, 1) * 100, 'b-', 'LineWidth', 5);
set(gca, 'LineWidth', 2, 'FontSize', 48);
legend('E\rightarrow{E}', 'E\rightarrow{I}', 'I\rightarrow{E}', 'I\rightarrow{I}');
xlabel('\delta (\mu{m})');
ylabel('p_{A,B}(\delta) (%)');
box off;
export_to_pdf('connection-prob-simulation.pdf');
%%
