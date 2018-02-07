function fExpectedRadius = ExpectedSpectrumRadius(fTotExc, fTotInh, nNumNeurons, fPropInh, fEFillFactor, fIFillFactor, nNumSSNs, fSSN)

if (numel(fEFillFactor) == 1)
   fEFillFactor(2) = fEFillFactor;
end

if (numel(fIFillFactor) == 1)
   fIFillFactor(2) = fIFillFactor;
end

fSSNW = fTotExc * fSSN * nNumSSNs / (fEFillFactor(1) * nNumNeurons) + (1-fSSN)*fTotExc/(fEFillFactor(1)*nNumNeurons);
fMeanSSNW = fTotExc * fSSN * nNumSSNs / nNumNeurons + (1-fSSN) * fTotExc / nNumNeurons;
fSSNVar = fEFillFactor(1)*(fSSNW - fMeanSSNW).^2 + (1-fEFillFactor(1)) * fMeanSSNW.^2;

fMeanEW = fTotExc/nNumNeurons;
fProbSSN = fEFillFactor(1)*(1-fPropInh)/nNumSSNs;
fNonSSNW = (1-fSSN) * fTotExc / (fEFillFactor(1) * nNumNeurons);
fProbNonSSN = fEFillFactor(1) * (1-fPropInh) * (1-1/nNumSSNs);
fEIW = fTotExc / (fEFillFactor(2) * nNumNeurons);
fProbEI =  fPropInh * fEFillFactor(2);
fExcVar = (1-fEFillFactor(1))*fMeanEW.^2 + fProbSSN * (fSSNW - fMeanEW).^2 + ...
          fProbNonSSN * (fNonSSNW - fMeanEW).^2 + fProbEI * (fEIW - fMeanEW).^2;

fInhMean = -fTotInh / nNumNeurons;
fInhVar = (1-fIFillFactor(1)) .* fPropInh .* fInhMean.^2 + ...
          (1-fIFillFactor(2)) .* (1-fPropInh) .* fInhMean.^2 + ...
          fIFillFactor(1) .* fPropInh .* (-fTotInh ./ (fIFillFactor(1) .* nNumNeurons) - fInhMean).^2 + ...
          fIFillFactor(2) .* (1-fPropInh) .* (-fTotInh ./ (fIFillFactor(2) .* nNumNeurons) - fInhMean).^2;


fExpectedRadius = sqrt(nNumNeurons*(fPropInh*fInhVar + (1-fPropInh)*fExcVar));

       