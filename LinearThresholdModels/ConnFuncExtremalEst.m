function [vfMeans, vfStds, fSii, vfDomain, vfLnMinSum, vfLnMaxSum] = ...
            ConnFuncExtremalEst(nNumNeurons, nL, fConnStd, nNumDims)
         
% ConnFuncExtremalEst - FUNCTION Estimate expectations of extremal distributions for the connectivity function G
%
% Usage: [vfMeans, vfStds, fSii, vfDomain, vfLnMinSum, vfLnMaxSum] = ...
%             ConnFuncExtremalEst(nNumNeurons, nL, fConnStd, nNumDims)
%
% Input arguments: LaTex equivalent defined in [1]
% nNumNeurons: N
% nL: L
% fConnStd: \kappa
% nNumDims: D
%
% Output arguments: LaTex equivalent defined in [1]
% vfMeans: [E[MinEst(\Sigma_L {S_{i\neq j}})] E[MaxEst(\Sigma_L {S_{i\neq j}})]]
% vfStds: [std(MinEst(\Sigma_L {S_{i\neq j}})) std(MaxEst(\Sigma_L {S_{i\neq j}}))]
% fSii: s_{ii}
%
% vfDomain: Vector of points over which extremal distributions of \Sigma_L {S_{i\neq j} wer sampled
% vfLnMinSum: Vector of samples of MinEst(\Sigma_L {S_{i\neq j}})), corresponding to 'vfDomain'
% vfLnMaxSum: Vector of samples of MaxEst(\Sigma_L {S_{i\neq j}})), corresponding to 'vfDomain'
%
% References: [1] Muir and Mrsic-Flogel. "Eigenspectrum bounds for
%     semi-random matrices with modular and spatial structure for neural
%     networks." In preparation.

% Author: Dylan Muir <dylan.muir@unibas.ch>
% Created: June 2014

fGaussNorm = nNumDims^nNumDims .* pi.^(nNumDims/2) * fConnStd.^nNumDims * erf(1/(2*nNumDims*fConnStd)).^nNumDims;
fGaussVar = (nNumDims/sqrt(2))^nNumDims * pi^(nNumDims/2) * fConnStd^nNumDims * erf(1/(nNumDims*fConnStd*sqrt(2)))^nNumDims - fGaussNorm^2;

% - Correction for tighter estimate of extremal distributions, to take
%      correlated structure of matrix S into account
% fGaussVar = fGaussVar / (1 + 2 / nNumDims);

fK = nL * fGaussNorm^2 / fGaussVar;
fTh = fGaussVar / ((nNumNeurons-1)*fGaussNorm^2 + fGaussNorm);

fMu = fK*fTh;
fVar = fK*fTh^2;

vfDomain = [linspace(fMu-sqrt(fVar)*5, fMu+sqrt(fVar)*5, 400) 0:.01:2];
vfDomain(vfDomain <= 0) = nan;

vfLnMinSum = (-vfDomain./fTh) - fK.*log(fTh) + log(nL) + (fK-1).*log(vfDomain) + (nL-1).*log(gammainc(vfDomain./fTh, fK,'upper')) - gammaln(fK);
vfLnMaxSum = (-vfDomain./fTh) - fK.*log(fTh) + log(nL) + (fK-1).*log(vfDomain) + (nL-1).*log(1-gammainc(vfDomain./fTh, fK, 'upper')) - gammaln(fK);

vfMeans(1) = nansum(exp(vfLnMinSum).*vfDomain)./nansum(exp(vfLnMinSum));
vfMeans(2) = nansum(exp(vfLnMaxSum).*vfDomain)./nansum(exp(vfLnMaxSum));

% - Catch expectations of zero
vfMeans(isnan(vfMeans)) = 0;

vfVars(1) = nansum(exp(vfLnMinSum).*(vfDomain-vfMeans(1)).^2);
vfVars(2) = nansum(exp(vfLnMaxSum).*(vfDomain-vfMeans(1)).^2);

vfStds = sqrt(vfVars);

fSii = 1/(fGaussNorm*(nNumNeurons-1) + 1);

% --- END of ConnFuncExtremalDist.m ---
