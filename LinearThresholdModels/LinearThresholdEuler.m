function [vfX_tplus, oState] = LinearThresholdEuler(oState, vfI_t, fDeltaT, pActivationParams)

% LinearThreshold - FUNCTION Linear-threshold activation function
%
% Usage: function [vfX_tplus, oState] = LinearThresholdEuler(oState, vfI_t, fDeltaT, pActivationParams)
%

% Author: Dylan Muir <dylan@ini.phys.ethz.ch>
% Created: 16th November, 2008

% -- Defaults

DEF_fGain = 1;
DEF_fThreshold = 0;
DEF_fTauNoiseSigma = 0;


% -- Check arguments

if (nargin < 4)
   disp('*** LinearThresholdEuler: Incorrect usage');
   help LinearThresholdEuler;
   return;
end


% -- Get parameters, reset state

nNumUnits = numel(vfI_t);

fGain = GetParam(pActivationParams, 'fGain', DEF_fGain);
fThreshold = GetParam(pActivationParams, 'fThreshold', DEF_fThreshold);

if (isempty(oState))
   fTau = GetParam(pActivationParams, 'fTau');
   fTauNoiseSigma = GetParam(pActivationParams, 'fTauNoiseSigma', DEF_fTauNoiseSigma);
   oState.vfActivation = vfI_t;%zeros(nNumUnits, 1);
   oState.vfTau = normrnd(fTau, fTauNoiseSigma, nNumUnits, 1);
   oState.vfTau(oState.vfTau < 0) = 1e-3;
   oState.fhGetState = @LTE_GetState;
end


% -- Compute Xdot(t), X(t+1)

vfXdot = 1./oState.vfTau .* (-oState.vfActivation + vfI_t(:));

oState.vfActivation = oState.vfActivation + fDeltaT * vfXdot;
oState.vfActivation(isnan(oState.vfActivation)) = inf;

vfX_tplus = fGain * reshape(oState.vfActivation - fThreshold, size(vfI_t));
vfX_tplus(vfX_tplus < 0) = 0;

return;

% -- DE and difference equations

% tau * x_dot + x = alpha * (sum(Wq * thresh(x_q - theta_q)) + i);

x_dot(t) = 1/tau * (-x(t) + alpha * (sum(Wq * thresh(x_q(t) - theta_q)) + i)); %#ok<UNRCH>

x(t+1) = x(t) + fDeltaT * x_dot(t);

function vfState = LTE_GetState(oState)
vfState = oState.vfActivation;


% --- END of LinearThreshold.m ---
