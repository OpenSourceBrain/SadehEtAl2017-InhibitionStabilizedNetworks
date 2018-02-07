fTotE = 5.4;
fTotI = 56;
fPropInh = 0.2;
fEConnFillFactor = [.0022 .072];
fIConnFillFactor = [.084 .34];
nNumSSNs = 1;
fSSN = 0;

vN = logspace(1, 7, 20);
clear vfExpectedRadius
for nIndex = 1:numel(vN)
   vfExpectedRadius(nIndex) = ExpectedSpectrumRadius(fTotE, fTotI, vN(nIndex), fPropInh, fEConnFillFactor, fIConnFillFactor, nNumSSNs, fSSN);
end

figure;
plot(vN, vfExpectedRadius);
hold all;
set(gca, 'XScale', 'log');
plot(xlim, [1 1], 'k:');


%%

N = 3300;

fhRadSearch = @(fS)abs(1-ExpectedSpectrumRadius(fTotE, fTotI, N, fPropInh, fS * fEConnFillFactor, fS * fIConnFillFactor, nNumSSNs, fSSN));
fSpecScale = fminsearch(fhRadSearch, 10)

fSpecScale.*fIConnFillFactor