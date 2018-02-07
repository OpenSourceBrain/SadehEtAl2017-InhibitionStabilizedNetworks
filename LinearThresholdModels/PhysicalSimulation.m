function [vfPertResp, vfPertWidths] = PhysicalSimulation(vfAxonWidths, vfDendWidths, fETot, fITot, fPerturbStrength, bCheckEigs, bPlot)

if (~exist('bPlot', 'var'))
   bPlot = true;
end

if (~exist('bCheckEigs', 'var') || isempty(bCheckEigs))
   bCheckEigs = false;
end

if (~exist('fPerturbStrength', 'var') || isempty(fPerturbStrength))
   fPerturbStrength = 0.8;
end


%% Resolution parameters

fGridPixelSize = 33e-6;
vfGridSize = 2400e-6 * [1 1];
nExcNodesPerGrid = 1;
nInhNodesPerGrid = 1;


%% Weight parameters

if (~exist('fETot', 'var') || isempty(fETot))
   fETot = 5.4;
end

if (~exist('fITot', 'var') || isempty(fITot))
   fITot = 5.4;
end

fInhProp = .5;


%% Physical dimension parameters

if (~exist('vfAxonWidths', 'var') || isempty(vfAxonWidths))
   fExcAxonWidth = sqrt(1200^2 - 300^2);
   vfAxonWidths = [fExcAxonWidth 300] * 1e-6;
end

if (~exist('vfDendWidths', 'var') || isempty(vfDendWidths))
   vfDendWidths = [300 300] * 1e-6;
end

fExcAxonWidth = vfAxonWidths(1);
fInhAxonWidth = vfAxonWidths(2);

fExcDendWidth = vfDendWidths(1);
fInhDendWidth = vfDendWidths(2);


%% Derived parameters

vnGridDims = round(vfGridSize ./ fGridPixelSize);
fGridPixelSize = mean(vfGridSize ./ vnGridDims);

vfX = linspace(0, vfGridSize(1), vnGridDims(1)+1) + fGridPixelSize/2;
vfX = vfX(1:end-1);
vfY = linspace(0, vfGridSize(2), vnGridDims(2)+1) + fGridPixelSize/2;
vfY = vfY(1:end-1);
[mfX, mfY] = ndgrid(vfX, vfY);
[mnX, mnY] = ndgrid(1:vnGridDims(1), 1:vnGridDims(2));

nNumExcNeurons = numel(mfX) * nExcNodesPerGrid;
nNumInhNeurons = numel(mfX) * nInhNodesPerGrid;
nNumNeurons =  nNumExcNeurons + nNumInhNeurons;
vnExcNeurons = 1:nNumExcNeurons;
vnInhNeurons = (1:nNumInhNeurons) + nNumExcNeurons;


%% Generate connectivity matrices

% - Compute distance mesh
mfDistMesh = reshape(TorusDistance2d(vfGridSize, fGridPixelSize /2 * [1 1], [mfX(:) mfY(:)]), vnGridDims);

% - Gaussian function
fhGauss = @(d, w)exp(-d.^2 ./ 2 ./ (w/4).^2);

% - Compute axon and dendrite meshes
mfEAxon = fhGauss(mfDistMesh, fExcAxonWidth);% exp(-mfDistMesh.^2 ./ 2 ./ (fExcAxonWidth/4).^2);
mfEAxon = mfEAxon ./ sum(mfEAxon(:));
mfIAxon = fhGauss(mfDistMesh, fInhAxonWidth);%exp(-mfDistMesh.^2 ./ 2 ./ (fInhAxonWidth/4).^2);
mfIAxon = mfIAxon ./ sum(mfIAxon(:));

mfEDend = fhGauss(mfDistMesh, fExcDendWidth);%exp(-mfDistMesh.^2 ./ 2 ./ (fExcDendWidth/4).^2);
mfEDend = mfEDend ./ sum(mfEDend(:));
mfIDend = fhGauss(mfDistMesh, fInhDendWidth);%exp(-mfDistMesh.^2 ./ 2 ./ (fInhDendWidth/4).^2);
mfIDend = mfIDend ./ sum(mfIDend(:));


% - Preallocate weight matrix
mfWeights = zeros(nNumNeurons);

% - Compute weights from each source location
for (nUnitSource = 1:numel(mfX))
   % - Find source unit location
   vnLoc = [mnX(nUnitSource) mnY(nUnitSource)];   
   
   % - Compute E to E connections
   mfEEDist = mfEAxon .* mfEDend;
   mfEEDist = mfEEDist ./ sum(mfEEDist(:)) ./ nExcNodesPerGrid;
   mfEEWeights = mfEEDist .* fETot .* (1-fInhProp);
   
   % - Compute E to I connections
   mfEIDist = mfEAxon .* mfIDend;
   mfEIDist = mfEIDist ./ sum(mfEIDist(:)) ./ nInhNodesPerGrid;
   mfEIWeights = mfEIDist .* fETot .* fInhProp;
   
   % - Compute I to E connections
   mfIEDist = mfIAxon .* mfEDend;
   mfIEDist = mfIEDist ./ sum(mfIEDist(:)) ./ nExcNodesPerGrid;
   mfIEWeights = mfIEDist .* -abs(fITot) .* (1-fInhProp);
   
   % - Compute I to I connections
   mfIIDist = mfIAxon .* mfIDend;
   mfIIDist = mfIIDist ./ sum(mfIIDist(:)) ./ nInhNodesPerGrid;
   mfIIWeights = mfIIDist .* -abs(fITot) .* fInhProp;
   
   % - Make weight tensor
   tfWeights = cat(3,   repmat(mfEEWeights, [1 1 nExcNodesPerGrid]), ...
                        repmat(mfEIWeights, [1 1 nInhNodesPerGrid]), ...
                        repmat(mfIEWeights, [1 1 nExcNodesPerGrid]), ...
                        repmat(mfIIWeights, [1 1 nInhNodesPerGrid]));
   tfWeights = circshift(tfWeights, [vnLoc-1 0]);
   
   % - Insert into weight matrix
   mfWeights(vnExcNeurons, vnExcNeurons(nUnitSource)) = reshape(tfWeights(:, :, 1:nExcNodesPerGrid), 1, []);
   mfWeights(vnInhNeurons, vnExcNeurons(nUnitSource)) = reshape(tfWeights(:, :, nExcNodesPerGrid + (1:nInhNodesPerGrid)), 1, []);
   mfWeights(vnExcNeurons, vnInhNeurons(nUnitSource)) = reshape(tfWeights(:, :, 3), 1, []);
   mfWeights(vnInhNeurons, vnInhNeurons(nUnitSource)) = reshape(tfWeights(:, :, 4), 1, []);
end


%% Plot axonal fields

% if (bPlot)
%    mfEAxonPlot = circshift(mfEAxon .* fETot, round(vnGridDims/2));
%    fAxonThreshold = min(0.1*(max(mfEAxonPlot(:))), 0.1*(max(mfIAxonPlot(:))));
%    mfEAxonPlot(mfEAxonPlot < fAxonThreshold) = nan;
%    vfShiftDist = round(vnGridDims/2) * fGridPixelSize + fGridPixelSize/2;
%    
%    figure;
%    surf((vfX-vfShiftDist(1))*1e6, (vfY-vfShiftDist(2))*1e6, mfEAxonPlot);
%    hold all;
%    
%    mfIAxonPlot = circshift(mfIAxon .* abs(fITot), round(vnGridDims/2));
%    mfIAxonPlot(mfIAxonPlot < fAxonThreshold) = nan;   
%    surf((vfX-vfShiftDist(1))*1e6, (vfY-vfShiftDist(2))*1e6, -mfIAxonPlot);
%    shading flat;
%    axis vis3d;
%    xlabel('{\mu}m');
%    ylabel('{\mu}m');
%    zlabel('Weight');
%    title('Axonal fields');
%    
%    figure;
%    surf((vfX-vfShiftDist(1))*1e6, (vfY-vfShiftDist(2))*1e6, circshift(mfEDend, round(vnGridDims/2)));
%    hold all;
%    surf((vfX-vfShiftDist(1))*1e6, (vfY-vfShiftDist(2))*1e6, circshift(-mfIDend, round(vnGridDims/2)));
%    axis vis3d;
%    shading flat;
%    xlabel('{\mu}m');
%    ylabel('{\mu}m');
%    zlabel('Weight');
%    title('Dendritic fields');
% end

%% Plot axonal and dendritic profiles

if bPlot
   vfFineMesh = linspace(-vfGridSize(1)/2, vfGridSize(1)/2, 1000);
   vfCoarseMesh = vfX - max(vfX)/2;
   fMaxE = max(mfEAxon(:));
   fMaxI = max(abs(mfIAxon(:)));
   
   figure;
   plot(vfFineMesh*1e6, fMaxE .* fhGauss(vfFineMesh, fExcAxonWidth), 'k', 'LineWidth', 1);
   hold all;
   plot(vfCoarseMesh*1e6, fMaxE .* fhGauss(vfCoarseMesh, fExcAxonWidth), '.', 'Color', [1 0 0], 'MarkerSize', 6);

   plot(vfFineMesh*1e6, -fMaxI .* fhGauss(vfFineMesh, fInhAxonWidth), 'k-', 'LineWidth', 1);
   plot(vfCoarseMesh*1e6, -fMaxI .* fhGauss(vfCoarseMesh, fInhAxonWidth), '.', 'MarkerSize', 6, 'Color', [0 .5 1]);   
   
   axis tight;
   set(gca, 'LineWidth', 1, 'Box', 'off', 'Units', 'centimeters', 'OuterPosition', [0 0 4 3], 'FontSize', 6);
   xlabel('Separation ({\mu}m)', 'FontSize', 8);
   ylabel('Axonal syn. weight (a.u.)', 'FontSize', 8);
   set(gca, 'YTick', 0);


   figure;
   plot(vfFineMesh*1e6, fhGauss(vfFineMesh, fExcDendWidth), 'k-', 'LineWidth', 1);
   hold all;
   plot(vfCoarseMesh*1e6, fhGauss(vfCoarseMesh, fExcDendWidth), '.', 'Color', [1 0 0], 'MarkerSize', 6);

   plot(vfFineMesh*1e6, -fhGauss(vfFineMesh, fInhDendWidth), 'k-', 'LineWidth', 1);
   plot(vfCoarseMesh*1e6, -fhGauss(vfCoarseMesh, fInhDendWidth), '.', 'Color', [0 .5 1], 'MarkerSize', 6);   
   
   axis tight;
   
   set(gca, 'LineWidth', 1, 'Box', 'off', 'Units', 'centimeters', 'OuterPosition', [0 0 4 3], 'FontSize', 6);
   xlabel('Separation ({\mu}m)', 'FontSize', 8);
   ylabel('Dendrite density (a.u.)', 'FontSize', 8);
   set(gca, 'YTick', 0);
end


%% Check stability

if bCheckEigs
try
vfTopEigs = eigs(mfWeights + 1e-4.*(rand(size(mfWeights))-.5), 2, 'lr');

if any(real(vfTopEigs) > 1)
   warning('Unstable network');
   vfPertResp = nan;
   vfPertWidths = nan;
   return;
end
catch
   warning('Eigenvalues failed');
end
end

%% Simulate perturbation and plot results

mfDistCentre = sqrt((mfX - vfGridSize(1)/2).^2 + (mfY - vfGridSize(2)/2).^2);
tSimTimeChunks = 6;

[~, nMinLoc] = min(mfDistCentre(:));
[nRowCentre, nColCentre] = ind2sub(size(mfDistCentre), nMinLoc);

nCentreExc = sub2ind([vnGridDims 2], nRowCentre, nColCentre, 1);
nCentreInh = sub2ind([vnGridDims 2], nRowCentre, nColCentre, 2);

vnExcRow = sub2ind([vnGridDims 2], nRowCentre * ones(1, vnGridDims(1)), 1:vnGridDims(2), ones(1, vnGridDims(1)));
vnInhRow = sub2ind([vnGridDims 2], nRowCentre * ones(1, vnGridDims(1)), 1:vnGridDims(2), 2 * ones(1, vnGridDims(1)));
   
if (bPlot)   
   fPerturbWidth = 600e-6;
   
   vbPerturbInh = (mfDistCentre(:) < fPerturbWidth/2)';
   
   vfInput = ones(1, nNumNeurons);
   vfAct = zeros(1, nNumNeurons);
   
   vtDuration = [0 tSimTimeChunks];
   
   XdTRec = @(vfX, Wrec, vfInput)Wrec * max(vfX(:), 0) + vfInput(:) - vfX(:);
   
   [t_Baseline, mfX_Baseline] = ode45(@(t, x)XdTRec(x, mfWeights, vfInput), vtDuration, vfAct);
   
   % - Provide perturbation
   vfInputPert = vfInput;
   vfInputPert(vnInhNeurons(vbPerturbInh)) = fPerturbStrength;
   
   vtPert = [0 tSimTimeChunks] + tSimTimeChunks;
   [t_Pert, mfX_Pert] = ode45(@(t, x)XdTRec(x, mfWeights, vfInputPert), vtPert, mfX_Baseline(end, :));
   
   % - Plot results
   
%    figure;
%    plot(t_Baseline, mfX_Baseline(:, vnExcNeurons), 'r-');
%    hold all;
%    plot(t_Baseline, mfX_Baseline(:, vnInhNeurons), 'b-');
%    plot(t_Pert, mfX_Pert(:, vnExcNeurons), 'r-');
%    if (~all(vbPerturbInh)), plot(t_Pert, mfX_Pert(:, vnInhNeurons(~vbPerturbInh)), 'b-'); end
%    if (any(vbPerturbInh)),  plot(t_Pert, mfX_Pert(:, vnInhNeurons(vbPerturbInh)), 'k-'); end
%    plot(xlim, [0 0], 'k:');
%    xlabel('Time (s)');
%    ylabel('Response (a.u.)');
   
   figure;
   plot((vfX-vfGridSize(1)/2)*1e6, mfX_Pert(end, vnInhRow)-mfX_Baseline(end, vnInhRow), 'LineWidth', 1, 'Color', [0 .5 1]);
   hold all;
   plot((vfX-vfGridSize(1)/2)*1e6, mfX_Pert(end, vnExcRow)-mfX_Baseline(end, vnExcRow), 'LineWidth', 1, 'Color', [1 0 0]);
   plot(xlim, [0 0], 'k:', 'LineWidth', 1);
   plot(fPerturbWidth/2 * [-1 1]*1e6, 0.8 * [1 1], 'k-', 'LineWidth', 2);
   set(gca, 'Box', 'off', 'FontSize', 6, 'LineWidth', 1, 'YTick', 0);
   set(gca, 'Units', 'centimeters', 'OuterPosition', [0 0 12 4]);
   xlabel('Dist. from center ({\mu}m)', 'FontSize', 8);
   ylabel('Response difference (a.u.)', 'FontSize', 8);
   ylim([-.5 1]);
   xlim(600*[-1 1])


   vfPertResponse = mfX_Pert(end, vnInhNeurons) - mfX_Baseline(end, vnInhNeurons);
   
   figure;
   subplot(1, 2, 1);
   imagesc(vfX, vfY, reshape(vfPertResponse, vnGridDims));
   hold all;
   % vfCAxis = caxis;
   % contour(mfX, mfY, reshape(vfPertResponse, vnGridDims), [0 0], 'k-');
   % caxis(vfCAxis);
   % colorbar;
   axis equal tight;
   xlabel('mm');
   ylabel('mm');
   title('Response \delta');
   
   vbParadoxical = ((vfPertResponse > 0) & (fPerturbStrength < 1)) | ((vfPertResponse < 0) & (fPerturbStrength > 1));
   
   subplot(1, 2, 2);
   
   imagesc(vfX, vfY, reshape((vbParadoxical & vbPerturbInh), vnGridDims));
   axis equal tight;
   xlabel('mm');
   ylabel('mm');
   title('Is paradoxical');
   
   fprintf('MaxMinPerturbedResp = [%.2f %.2f]\n', min(vfPertResponse(vbPerturbInh)), max(vfPertResponse(vbPerturbInh)));
   
end

%% Simulate range of perturbation widths

vfPertWidths = linspace(min(mfDistCentre(:))*2, 250e-6, 20);
[~, nCentreInh] = min(mfDistCentre(:));

clear vfMinPertResp vfMaxPertResp;
for (nPertIndex = 1:numel(vfPertWidths))
   
   fprintf('Pert. diameter [%d / %d]\n', nPertIndex, numel(vfPertWidths));
   
   fPerturbWidth = vfPertWidths(nPertIndex);
   vbPerturbInh = (mfDistCentre(:) <= fPerturbWidth/2)';
   
   vfInput = ones(1, nNumNeurons);
   vfAct = zeros(1, nNumNeurons);
   
   vtDuration = [0 tSimTimeChunks];
   
   XdTRec = @(vfX, Wrec, vfInput)Wrec * max(vfX(:), 0) + vfInput(:) - vfX(:);
   
   [~, mfX_Baseline] = ode45(@(t, x)XdTRec(x, mfWeights, vfInput), vtDuration, vfAct);
   
   % - Provide perturbation
   vfInputPert = vfInput;
   vfInputPert(vnInhNeurons(vbPerturbInh)) = fPerturbStrength;
   
   vtPert = [0 tSimTimeChunks] + tSimTimeChunks;
   [~, mfX_Pert] = ode45(@(t, x)XdTRec(x, mfWeights, vfInputPert), vtPert, mfX_Baseline(end, :));

   vfPertResponse = (mfX_Pert(end, vnInhNeurons) - mfX_Baseline(end, vnInhNeurons)) ./ mfX_Baseline(end, vnInhNeurons);
   
   % - Record perturbation response
   vfPertResp(nPertIndex) = vfPertResponse(nCentreInh);
end

%% - Plot results


if (bPlot)
   figure;
   plot(vfPertWidths/1e-3, vfPertResp);
   hold all;
   plot(vfPertWidths([1 end])/1e-3, [0 0], 'k:');
   xlabel('Perturbation dia. ({\mu}m)');
   ylabel('Perturbatiion \delta resp. (a.u.)');
end
