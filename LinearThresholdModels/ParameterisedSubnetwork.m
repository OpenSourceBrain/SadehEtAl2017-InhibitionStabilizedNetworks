function [mfWeights, sMetrics, mfConnProb, vnSSNIDs] = ...
   ParameterisedSubnetwork(nNumNeurons, fPropInh, nNumSSNs, fSSN, ...
                           fConnParam, nSpaceDimensionality, ...
                           fTotExc, fTotInh, fEFillFactor, fIFillFactor)

% ParameterisedSubnetwork - FUNCTION Construct a random network connection matrix, with subnetwork structure
%
% Usage: [mfWeights, sMetrics, mfConnProb, vnSSNIDs] = ...
%    ParameterisedSubnetwork(nNumNeurons, <fPropInh, nNumSSNs, fSSN, ...
%                            fConnParam, nSpaceDimensionality, ...
%                            fTotExc, fTotInh, fEFillFactor, fIFillFactor>)
%
% Generates a random sparse connection matrix, with defined and tunable
% subnetwork structure. The design of the matrix is described in [1].
%
% Parameters: (LaTex equivalent as defined in [1])
% nNumNeurons: N
% fPropInh: f_I
% nNumSSNs: M
% fSSN: r
% fConnParam: \kappa
% nSpaceDimensionality: D
% fTotExc: w_E (positive)
% fTotInh: w_I (positive)
% fEFillFactor: h_E
% fIFillFactor: h_I
%
% Note that fEFillFactor and fIFillFactor can be provided as vectors
% [fEEFillFactor fEIFillFactor] and [fIIFillFactor fIEFillFactor].
%
% Output arguments: (LaTex equivalent as defined in [1])
% mfWeights: W
% sMetrics structure:
% sMetrics.fSSNRadius: \sigma_Q
% sMetrics.fExpectedRadius: \sqrt(N*(f_I*\sigma_I + (1-f_I)*\sigma_E))
% sMetrics.fCentralRadius: \min(\sqrt(N* [{\sigma_E}^2 {\sigma_I}^2]))
% sMetrics.fPredTrivialEigenvalue: \lambda_b
% sMetrics.fPredSSNEigenvalue: \lambda_Q
% 
% sMetrics.fSSNVar: {\sigma_Q}^2
% sMetrics.fExcVar: {\sigma_E}^2
% sMetrics.fInhVar: {\sigma_I}^2
% 
% sMetrics.fGaussNorm: E[s] Eqn(11)
% sMetrics.fPredPartitionESize: E[s] * N * (1-f_I)
% sMetrics.fPredPartitionISize: E[s] * N * f_I
% sMetrics.fPredNumPartitions: 1/E[s]
%
% sMetrics.vfMaxSpatialEigPartitionEst: [MaxEst(\hat{w}_ee) MinEst(\hat{w}_ii) MaxEst(\hat{w}_ie) MinEst(\hat{w}_ei)]
% sMetrics.fPredMaxSpatialEig: Maximal estimate for \lambda_+
%
% sMetrics.vfMinSpatialEigPartitionEst: [MinEst(\hat{w}_ee) MaxEst(\hat{w}_ii) MinEst(\hat{w}_ie) MaxEst(\hat{w}_ei)]
% sMetrics.fPredMinSpatialEig: Minimal estimate for lambda_-
%
% References: [1] Muir and Mrsic-Flogel. 2015 "Eigenspectrum bounds for
%     semi-random matrices with modular and spatial structure for neural
%     networks." PRE.

% Author: Dylan Muir <dylan.muir@unibas.ch>
% Created: June 2014

% -- Defaults

DEF_fPropInh = 0.2;           % - By default 20% inhibitory neurons
DEF_nNumSSNs = 1;             % - By default, no effect of simple SSN partitions
DEF_fSSN = 0;                 % - By default, no effect of simple SSN partitions
DEF_fConnParam = inf;         % - By default, no effect of spatial connection specificity
DEF_nSpaceDimensionality = 0; % - By default, no effect of spatial connection specificity
DEF_fTotExc = 5;              % - By default, construct an inhibition-stabilised network
DEF_fTotInh = 10;             % - By default, construct an inhibition-stabilised network
DEF_fEFillFactor = 1;         % - By default, the network connectivity is full
DEF_fIFillFactor = 1;         % - By default, the network connectivity is full


% -- Check arguments

if (nargin == 0)
   help ParameterisedSubnetwork;
   error('*** ParameterisedSubnetwork: Incorrect usage');
end

if (~exist('fPropInh', 'var') || isempty(fPropInh))
   fPropInh = DEF_fPropInh;
end
   
if (~exist('nNumSSNs', 'var') || isempty(nNumSSNs))
   nNumSSNs = DEF_nNumSSNs;
end

if (~exist('fSSN', 'var') || isempty(fSSN))
   fSSN = DEF_fSSN;
end

if (~exist('fConnParam', 'var') || isempty(fConnParam))
   fConnParam = DEF_fConnParam;
end

if (~exist('nSpaceDimensionality', 'var') || isempty(nSpaceDimensionality))
   nSpaceDimensionality = DEF_nSpaceDimensionality;
end

if (~exist('fTotExc', 'var') || isempty(fTotExc))
   fTotExc = DEF_fTotExc;
end

if (~exist('fTotInh', 'var') || isempty(fTotInh))
   fTotInh = DEF_fTotInh;
end

if (~exist('fEFillFactor', 'var') || isempty(fEFillFactor))
   fEFillFactor = DEF_fEFillFactor;
end

if (numel(fEFillFactor) < 2)
   fEFillFactor(2) = fEFillFactor(1);
end

if (~exist('fIFillFactor', 'var') || isempty(fIFillFactor))
   fIFillFactor = DEF_fIFillFactor;
end

if (numel(fIFillFactor) < 2)
   fIFillFactor(2) = fIFillFactor(1);
end

% - Ensure fTotInh is positive
fTotInh = abs(fTotInh);

% - Consolidate connectivity function parameters
fConnStd = fConnParam;
fConnKappa = fConnParam;

% - Work out how many neurons in network, to ensure round numbers and
% correct proportions
nNumNeuronsPerSSN = round(nNumNeurons * (1-fPropInh) / nNumSSNs);
nNumExc = nNumNeuronsPerSSN * nNumSSNs;

% - Incorporate inhibitory neurons
if (fPropInh == 1)
   nNumInh = nNumNeurons;
elseif (fPropInh == 0)
   nNumInh = 0;
else
   nNumInh = round(nNumExc / (1-fPropInh) * fPropInh);
end

nNumNeurons = nNumExc + nNumInh;

% - Draw random node locations
mfLocation = rand(nNumNeurons, nSpaceDimensionality);

% - Assign neurons to SSN partitions
vnSSNIDs = repmat((1:nNumSSNs), nNumNeuronsPerSSN, 1);
vnSSNIDs = vnSSNIDs(:);


% -- Use a multi-dimensional periodic von Mises-like function for F
% mfConnProb = squareform(pdist(mfLocation, @(xi, xj)(prod(exp((1/fConnKappa) * cos(2*pi * bsxfun(@minus, xj, xi))), 2))));
% % - Assign diagonal of spatial connection measure
% mfConnProb(logical(eye(nNumNeurons))) = exp((1/fConnKappa) * cos(0));

% -- Use a multi-dimensional periodic Gaussian function for F
mfConnProb = squareform(pdist(mfLocation, @(xi, xj)exp(-(sum((acos(cos(2*pi * bsxfun(@minus, xj, xi)))./(2*pi)).^2, 2)./(nSpaceDimensionality.^2 * fConnStd.^2)))));

% - Deal with the case of a network with no spatial/functional influence
if (nSpaceDimensionality < 1) || isinf(fConnStd)
   mfConnProb = ones(nNumNeurons);
   fGaussNorm = 1;

else  
   % - Assign diagonal of spatial connection measure
   mfConnProb(logical(eye(nNumNeurons))) = 1;
   
   fGaussNorm = nSpaceDimensionality^nSpaceDimensionality .* pi.^(nSpaceDimensionality/2) * fConnStd.^nSpaceDimensionality * erf(1/(2*nSpaceDimensionality*fConnStd)).^nSpaceDimensionality;
   % fGaussVar = (nSpaceDimensionality/sqrt(2))^nSpaceDimensionality * pi^(nSpaceDimensionality/2) * fConnStd^nSpaceDimensionality * erf(1/(nSpaceDimensionality*fConnStd*sqrt(2)))^nSpaceDimensionality - fGaussNorm^2;
end

fColNorm = fGaussNorm*(nNumNeurons-1) + 1;
fAutapseProp = 1 / fColNorm;

% - Normalise connection probability matrix (matrix S in [1])
mfConnProb = mfConnProb ./ fColNorm;

% -- Construct SSN connection matrix (matrix \mathbf{Q} in [1])
mfSameSSNConnProb = zeros(nNumNeurons);
for (nNeuron = 1:nNumExc)
   nThisSSN = vnSSNIDs(nNeuron);
   mfSameSSNConnProb(1:nNumExc, nNeuron) = double((vnSSNIDs == nThisSSN)) ./ nNumNeuronsPerSSN * (1-fPropInh);
end


% -- Compute nominal E and I weights (matrix W in [1])

mfExc = vertcat(mfConnProb(1:nNumExc, 1:nNumExc) * (1-fSSN) + mfSameSSNConnProb(1:nNumExc, 1:nNumExc) * fSSN, ...
                mfConnProb(nNumExc+1:end, 1:nNumExc));
mfInh = mfConnProb(:, nNumExc+1:end);


% -- Assign sparsity (effect of h_E and h_I in [1])

nNumSSNnz = round(nNumNeuronsPerSSN * fEFillFactor(1));
nNumENonSSNnz = round((nNumExc-nNumNeuronsPerSSN) * fEFillFactor(1));
% nNumEEnz = round(nNumExc * fEFillFactor);
nNumEInz = round(nNumInh * fEFillFactor(2));
nNumEnz = nNumSSNnz + nNumENonSSNnz + nNumEInz;
nNumIInz = round(nNumInh * fIFillFactor(1));
nNumIEnz = round(nNumExc * fIFillFactor(2));
nNumInz = nNumIEnz + nNumIInz;

% - Assign excitatory sparsity
if any(fEFillFactor < 1)
   vnInh = (nNumExc+1):nNumNeurons;
   for (nNeuron = 1:nNumExc)
      % - Find within-SSN, excitatory, inhibitory weights for this neuron
      vnInSSN = find(vnSSNIDs(nNeuron) == vnSSNIDs)';
      vnNonSSNExc = find(vnSSNIDs(nNeuron) ~= vnSSNIDs)';
      
      % - Assign SSN zeros
      % - Protect autapse
%       vnZeros = random_order(setdiff(vnInSSN, nNeuron));
%       mfExc(vnZeros(1:end-nNumSSNnz+1), nNeuron) = 0;
%       mfExc(vnZeros(end-nNumSSNnz+2:end), nNeuron) = mfExc(vnZeros(end-nNumSSNnz+2:end), nNeuron) ./ fEFillFactor(1);
%       mfExc(nNeuron, nNeuron) = mfExc(nNeuron, nNeuron) ./fEFillFactor;
% 
%       connprob = (1-fSSN) * (1-fPropInh) * (1-(fAutapseProp/(1-fPropInh)))/(nNumExc-1);
%       ssn = fSSN * (1-fPropInh) / nNumNeuronsPerSSN;
%       column = (connprob+ssn) * nNumNeuronsPerSSN * (nNumSSNnz-1) / nNumSSNnz + connprob * (nNumExc-nNumNeuronsPerSSN);
%       mfExc(nNeuron, nNeuron) = (1-fPropInh) - column;
      
      % - Do not protect autapse
      vnZeros = random_order(vnInSSN);
      mfExc(vnZeros(1:end-nNumSSNnz), nNeuron) = 0;
      mfExc(vnZeros(end-nNumSSNnz+1:end), nNeuron) = mfExc(vnZeros(end-nNumSSNnz+1:end), nNeuron) ./ fEFillFactor(1);
      
      % - Assign non-SSN zeros
      if (nNumSSNs > 1)
         vnZeros = random_order(vnNonSSNExc);
         mfExc(vnZeros(1:end-nNumENonSSNnz), nNeuron) = 0;
         mfExc(vnZeros(end-nNumENonSSNnz+1:end), nNeuron) = mfExc(vnZeros(end-nNumENonSSNnz+1:end), nNeuron) ./ fEFillFactor(1);
      end
      
      % - Assign E-I zeros
      vnZeros = random_order(vnInh);
      mfExc(vnZeros(1:end-nNumEInz), nNeuron) = 0;
      mfExc(vnZeros(end-nNumEInz+1:end), nNeuron) = mfExc(vnZeros(end-nNumEInz+1:end), nNeuron) ./ fEFillFactor(2);
   end
end

% - Assign inhibitory sparsity
if any(fIFillFactor < 1)
   vnExc = 1:nNumExc;
   vnInh = (nNumExc+1):nNumNeurons;
   for (nNeuron = 1:nNumInh)
      % - Assign I-E zeros
      vnZeros = random_order(vnExc);
      mfInh(vnZeros(1:end-nNumIEnz), nNeuron) = 0;
      mfInh(vnZeros(end-nNumIEnz+1:end), nNeuron) = mfInh(vnZeros(end-nNumIEnz+1:end), nNeuron) ./ fIFillFactor(2);
      
      % - Assign I-I zeros
      % - Protect autapse
%       vnZeros = random_order(setdiff(vnInh, vnInh(nNeuron)));
%       mfInh(vnZeros(1:end-nNumIInz+1), nNeuron) = 0;
%       mfInh(vnZeros(end-nNumIInz+2:end), nNeuron) = mfInh(vnZeros(end-nNumIInz+2:end), nNeuron) ./ fIFillFactor(1);
%       
%       connprob = (1-fAutapseProp)/(nNumNeurons-1)./fIFillFactor;
%       column = connprob * (nNumIEnz + nNumIInz - 1);
%       
%       mfInh(vnInh(nNeuron), nNeuron) = 1-column;

      % - Do not protect autapse
      vnZeros = random_order(vnInh);
      mfInh(vnZeros(1:end-nNumIInz), nNeuron) = 0;
      mfInh(vnZeros(end-nNumIInz+1:end), nNeuron) = mfInh(vnZeros(end-nNumIInz+1:end), nNeuron) ./ fIFillFactor(1);
   end
end

% - Randomly assign sparsity
% mfExc = mfExc .* double(rand(nNumNeurons, nNumExc) <= fEFillFactor);
% mfInh = mfInh .* double(rand(nNumNeurons, nNumInh) <= fIFillFactor);

% % - Equalise weights
% mfExc = bsxfun(@rdivide, mfExc, sum(mfExc));
% mfInh = bsxfun(@rdivide, mfInh, sum(mfInh));

% - Compose weight matrix
mfWeights = [fTotExc .* mfExc -fTotInh .* mfInh];

% - Estimate SSN and non-SSN eigenvalue spectrum radius
mbSSNWeights = false(nNumExc);
for (nSSN = 1:nNumSSNs)
   mbSSNWeights(vnSSNIDs == nSSN, vnSSNIDs == nSSN) = true;
end

mfEE = mfExc(1:nNumExc, :);

fSSNW = fTotExc * fSSN * nNumSSNs / (fEFillFactor(1) * nNumNeurons) + (1-fSSN)*fTotExc/(fEFillFactor(1)*nNumNeurons);
fMeanSSNW = fTotExc * fSSN * nNumSSNs / nNumNeurons + (1-fSSN) * fTotExc / nNumNeurons;
fSSNVar = fEFillFactor(1)*(fSSNW - fMeanSSNW).^2 + (1-fEFillFactor(1)) * fMeanSSNW.^2;

% fprintf(1, 'SSN variance: analytical: %6e; empirical: %6e\n', fSSNVar, var(mfEE(mbSSNWeights)));

% - Variance for single EI and EE fill-factor
% fMeanEW = fTotExc/nNumNeurons;
% fProbSSN = fEFillFactor(1)*(1-fPropInh)/nNumSSNs;
% fNonSSNW = (1-fSSN) * fTotExc / (fEFillFactor(1) * nNumNeurons);
% fProbNonSSN = fEFillFactor(1) * (1-fPropInh) * (1-1/nNumSSNs);
% fEIW = fTotExc / (fEFillFactor(1) * nNumNeurons);
% fProbEI =  fPropInh * fEFillFactor(1);
% fExcVar = (1-fEFillFactor(1))*fMeanEW.^2 + fProbSSN * (fSSNW - fMeanEW).^2 + ...
%           fProbNonSSN * (fNonSSNW - fMeanEW).^2 + fProbEI * (fEIW - fMeanEW).^2;

fMeanEW = fTotExc/nNumNeurons;
fProbSSN = fEFillFactor(1)*(1-fPropInh)/nNumSSNs;
fNonSSNW = (1-fSSN) * fTotExc / (fEFillFactor(1) * nNumNeurons);
fProbNonSSN = fEFillFactor(1) * (1-fPropInh) * (1-1/nNumSSNs);
fEIW = fTotExc / (fEFillFactor(2) * nNumNeurons);
fProbEI =  fPropInh * fEFillFactor(2);
fExcVar = (1-fEFillFactor(1))*fMeanEW.^2 + fProbSSN * (fSSNW - fMeanEW).^2 + ...
          fProbNonSSN * (fNonSSNW - fMeanEW).^2 + fProbEI * (fEIW - fMeanEW).^2;

% fprintf(1, 'Exc variance: analytical: %6e; empirical: %6e\n', fExcVar, var(mfExc(:)));

% - Variance for single IE and II fill-factor
% fInhVar = (1-fIFillFactor(1)) * (-fTotInh / nNumNeurons).^2 + ...
%           fIFillFactor(1) * (-fTotInh / (fIFillFactor(1) * nNumNeurons) + fTotInh/nNumNeurons).^2;

fInhMean = -fTotInh / nNumNeurons;
fInhVar = (1-fIFillFactor(1)) .* fPropInh .* fInhMean.^2 + ...
          (1-fIFillFactor(2)) .* (1-fPropInh) .* fInhMean.^2 + ...
          fIFillFactor(1) .* fPropInh .* (-fTotInh ./ (fIFillFactor(1) .* nNumNeurons) - fInhMean).^2 + ...
          fIFillFactor(2) .* (1-fPropInh) .* (-fTotInh ./ (fIFillFactor(2) .* nNumNeurons) - fInhMean).^2;
       
% fprintf(1, 'Inh variance: analytical: %6e; empirical: %6e\n', fInhVar, var(mfInh(:)));

if (nNumExc == 0)
   fExcVar = 0;
   fTotExc = 0;
end

if (nNumInh == 0)
   fInhVar = 0;
   fTotInh = 0;
end

% -- Estimate eigenvalue spectra
fSSNRadius = sqrt(fSSNVar);
fExpectedRadius = sqrt(nNumNeurons*(fPropInh*fInhVar + (1-fPropInh)*fExcVar));
fCentralRadius = min(sqrt(nNumNeurons * [fExcVar fInhVar]));
fPredTrivialEigenvalue = fTotExc*(1-fPropInh) - fTotInh*fPropInh;
fPredSSNEigenvalue = fTotExc * (1 - fPropInh) * fSSN;

fPredPartitionESize = fGaussNorm * nNumExc;
fPredPartitionISize = fGaussNorm * nNumInh;
fPredNumPartitions = 1/fGaussNorm;

if (fCentralRadius == 0)
   fCentralRadius = nan;
end

sMetrics.fSSNRadius = fSSNRadius;
sMetrics.fExpectedRadius = fExpectedRadius;
sMetrics.fCentralRadius = fCentralRadius;
sMetrics.fPredTrivialEigenvalue = fPredTrivialEigenvalue;
sMetrics.fPredSSNEigenvalue = fPredSSNEigenvalue;

sMetrics.fSSNVar = fSSNVar;
sMetrics.fExcVar = fExcVar;
sMetrics.fInhVar = fInhVar;

sMetrics.fGaussNorm = fGaussNorm;
sMetrics.fPredPartitionESize = fPredPartitionESize;
sMetrics.fPredPartitionISize = fPredPartitionISize;
sMetrics.fPredNumPartitions = fPredNumPartitions;


% -- Estimate extremal value partition eigenvalues

if (fPropInh < 1)
   [sSelf.vfMeantoE, sSelf.vfStdtoE, fSii] = ConnFuncExtremalEst(nNumNeurons, nNumNeurons * (1-fPropInh) -1, fConnStd, nSpaceDimensionality);
   [sOther.vfMeantoE, sOther.vfStdtoE] = ConnFuncExtremalEst(nNumNeurons, nNumNeurons * (1-fPropInh), fConnStd, nSpaceDimensionality);
else
   sSelf.vfMeantoE = [0 0];
   sSelf.vfStdtoE = [0 0];
   sOther.vfMeantoE = [0 0];
   sOther.vfStdtoE = [0 0];
   fSii = 0;
end

if (fPropInh > 0)
   [sSelf.vfMeantoI, sSelf.vfStdtoI] = ConnFuncExtremalEst(nNumNeurons, nNumNeurons * fPropInh -1, fConnStd, nSpaceDimensionality);
   [sOther.vfMeantoI, sOther.vfStdtoI] = ConnFuncExtremalEst(nNumNeurons, nNumNeurons * fPropInh, fConnStd, nSpaceDimensionality);
else
   sSelf.vfMeantoI = [0 0];
   sSelf.vfStdtoI = [0 0];
   sOther.vfMeantoI = [0 0];
   sOther.vfStdtoI = [0 0];
end

% - Estimate maximum eigenvalue
wee = (sSelf.vfMeantoE(2) + fSii) * fTotExc;    % MaxEst
wii = (sSelf.vfMeantoI(1) + fSii) * fTotInh;    % MinEst

wie = (sOther.vfMeantoI(2)) * fTotExc;          % MaxEst
wei = (sOther.vfMeantoE(1)) * fTotInh;          % MinEst

sMetrics.vfMaxSpatialEigPartitionEst = [wee wii wie wei];
sMetrics.fPredMaxSpatialEig = (1/2).*(sqrt((wee + wii).^2 - 4*wei.*wie) + wee - wii);

% - Estimate minimum eigenvalue
wee = (sSelf.vfMeantoE(1) + fSii) * fTotExc;    % MinEst
wii = (sSelf.vfMeantoI(2) + fSii) * fTotInh;    % MaxEst

wie = (sOther.vfMeantoI(1)) * fTotExc;          % MinEst
wei = (sOther.vfMeantoE(2)) * fTotInh;          % MaxEst

sMetrics.vfMinSpatialEigPartitionEst = [wee wii wie wei];
sMetrics.fPredMinSpatialEig = (1/2)*(-sqrt((wee + wii).^2 - 4*wei.*wie) + wee - wii);

% --- END of ParameterisedSubnetwork.m ---
