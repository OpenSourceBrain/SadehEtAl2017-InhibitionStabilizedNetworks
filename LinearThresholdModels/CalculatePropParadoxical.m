function [fStimDeriv, fNonStimDeriv, fPropParadoxical, fExcRespChangeProp, fStimInhRespChangeProp, vfInhInputBase, vfInhInputBoost] = CalculatePropParadoxical(nNumStim, nNumInh, sNetwork, fBoostDelta, fPropRand, bFPSolver)

fTOL = 1e-2;
tMaxTime = 6400e-3;

if (~exist('bFPSolver', 'var'))
   bFPSolver = true;
end

if (~exist('fPropRand', 'var') || isempty(fPropRand))
   fPropRand = 0;
end

nNumNeurons = size(sNetwork.mfWeights, 1);
nNumExc = nNumNeurons - nNumInh;

nNumStim = round(nNumStim);
if (nNumStim > nNumInh) || (nNumStim < 1)
   fStimDeriv = nan;
   fNonStimDeriv = nan;
   fPropParadoxical = nan;
   vfInhInputBase = nan;
   vfInhInputBoost = nan;
   fExcRespChangeProp = nan;
   fStimInhRespChangeProp = nan;
   vfBaseResp = nan;
   vfBoostResp = nan;
   return;
end

vfInputBase = (1-fPropRand) .* ones(nNumNeurons, 1) + fPropRand .* rand(nNumNeurons, 1);
vfInputBoost = fBoostDelta .* [zeros(nNumNeurons-nNumStim, 1); ones(nNumStim, 1)];

if (bFPSolver)
   fhFPResp = @(t, y, vfI)sum(abs(sNetwork.mfWeights * y - y + vfI))./nNumNeurons;
   
   options = optimoptions('fminunc', 'Display', 'none', 'Algorithm', 'Quasi-Newton');
   
   vfBaseResp = fminunc(@(x)fhFPResp(0, x, vfInputBase), vfInputBase .* nNumNeurons + rand(nNumNeurons, 1), options);
   vfBoostResp = fminunc(@(x)fhFPResp(0, x, vfInputBase + vfInputBoost), vfBaseResp - vfInputBoost .* rand(nNumNeurons, 1), options);
   
else  % - Use Euler solver
   tTimeStep = 1e-3;
   
   vfCurrentState = zeros(size(vfInputBase));
   tCurrentDur = 50e-3;
   
   while true
      [~, ~, ~, ~, mfResponseTrace, mfThisStateTrace] = ...
         SimulateNetwork(sNetwork, [], tTimeStep, tCurrentDur, @(~,~,~)vfInputBase, false, [1:nNumNeurons], vfCurrentState);
      vfCurrentState = mfThisStateTrace(:, end);
      vfBaseResp = mfResponseTrace(:, end);
      
      % - Test for fixed point
      if nanmean(abs(diff(mfThisStateTrace(:, end-1:end), [], 2))) < fTOL
         % - We can finish here
         break;
      else
         % - Continue from current point, with longer duration
         tCurrentDur = tCurrentDur * 2;
         
         if (tCurrentDur > tMaxTime)
            % - Give up
            vfBaseResp = nan(size(vfBaseResp));
            disp('GAVE UP');
            break;
         end
         
         continue;
      end
   end
   
%    figure;
%    plot(mean(mfResponseTrace(1:nNumExc, :), 1)', 'r-');
%    hold all;
%    plot(mean(mfResponseTrace(nNumExc+1:end-nNumStim-1, :), 1)', 'g--');
%    plot(mean(mfResponseTrace(end-nNumStim:end, :), 1)', 'b-');
%    % plot(mean(mfInhInput, 1)', ':');
%    legend('Exc', 'I non-stim', 'I stim', 'I total input');
% %    title(sprintf('W_E=%.2f; W_I=%.2f; f_I=%.2f; %.0f%% inh stim; %.2f boost', fTotExc, fTotInh, fPropInh, fPropInhStim*100, fBoostAmp));
   
   while true
      [~, ~, ~, ~, mfResponseTrace, mfThisStateTrace] = ...
         SimulateNetwork(sNetwork, [], tTimeStep, tCurrentDur, @(~,~,~)vfInputBase + vfInputBoost, false, [1:nNumNeurons], vfCurrentState);
      vfCurrentState = mfThisStateTrace(:, end);
      vfBoostResp = mfResponseTrace(:, end);
      
      % - Test for fixed point
      if nanmean(abs(diff(mfThisStateTrace(:, end-1:end), [], 2))) < fTOL
         % - We can finish here
         break;
      else
         % - Continue from current point, with longer duration
         tCurrentDur = tCurrentDur * 2;
         
         if (tCurrentDur > tMaxTime)
            % - Give up
            vfBoostResp = nan(size(vfBoostResp));
            disp('GAVE UP');
            break;
         end
         
         continue;
      end
   end
end

%    figure;
%    plot(mean(mfResponseTrace(1:nNumExc, :), 1)', 'r-');
%    hold all;
%    plot(mean(mfResponseTrace(nNumExc+1:end-nNumStim-1, :), 1)', 'g--');
%    plot(mean(mfResponseTrace(end-nNumStim:end, :), 1)', 'b-');
%    % plot(mean(mfInhInput, 1)', ':');
%    legend('Exc', 'I non-stim', 'I stim', 'I total input');
% %    title(sprintf('W_E=%.2f; W_I=%.2f; f_I=%.2f; %.0f%% inh stim; %.2f boost', fTotExc, fTotInh, fPropInh, fPropInhStim*100, fBoostAmp));

% - Estimate inhibitory input over time
vfInhInputBase = sNetwork.mfWeights(:, nNumExc+1:end) * max(vfBaseResp(nNumExc+1:end, :), 0);
vfInhInputBoost = sNetwork.mfWeights(:, nNumExc+1:end) * max(vfBoostResp(nNumExc+1:end, :), 0);

fPropParadoxical = nnz(-sign(fBoostDelta) .* ((vfInhInputBoost - vfInhInputBase) ./ vfInhInputBase) > 0.1) ./ numel(vfInhInputBase);

vfStimDiff = vfBoostResp - vfBaseResp;

fStimDeriv = nanmean(vfStimDiff(end-nNumStim+1:end)) ./ fBoostDelta;
fNonStimDeriv = nanmean(vfStimDiff(1:end-nNumInh)) ./ fBoostDelta;

fExcRespChangeProp = nanmean(vfStimDiff(1:nNumExc) ./ nanmean(vfBaseResp(1:nNumExc)));
fStimInhRespChangeProp = nanmean(vfStimDiff(nNumNeurons-nNumStim+1:end) ./ nanmean(vfBaseResp(nNumNeurons-nNumStim+1:end)));

% figure(101);
% hold on;
% plot(nNumStim/nNumInh*100, fStimDeriv, 'kx');

return;

%%

tTimeStep = 1e-3;
tStimDelay = 100e-3;

fhInputFunction = @(sN, tT, tDelta)vfInputBase + vfInputBoost .* (tT > tStimDelay);
fhMonitorFunction = @(varargin)MonitorTrace(varargin{:}, [1 nNumNeurons]);

[~, ~, ~, ~, ~, mfStateTrace] = ...
   SimulateNetwork(sNetwork, [], tTimeStep, tStimDelay*2, fhInputFunction, false, [1 nNumNeurons]);
nStimSteps = size(mfStateTrace, 2);
vfBaseResp = mfStateTrace(:, floor(nStimSteps/2));
vfBoostResp = mfStateTrace(:, nStimSteps);
vfStimDiff = vfBoostResp - vfBaseResp;

bBelowThresh = any(vfBoostResp <= 0);
