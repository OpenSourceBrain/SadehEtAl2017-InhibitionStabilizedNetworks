function MonitorTrace(sNetwork, ...
                        ctfActivity, coState, ...
                        vfTotalInput, vfExternalInput, ...
                        tTime, tTimeStep, hMonitorFigure, vnTraceIndices)
                   
persistent mfActivityTraces mfStateTraces vtTime;

if (tTime == 0)
   mfActivityTraces = [];
   mfStateTraces = [];
   vtTime = [];
end

vtTime = [vtTime tTime];
vfActivity = cell2mat(cellfun(@(c)reshape(c, [], 1), ctfActivity, 'UniformOutput', false));
mfActivityTraces(end+1, :) = vfActivity(vnTraceIndices);
vfStates = cell2mat(cellfun(@(c)c.fhGetState(c), coState, 'UniformOutput', false));
mfStateTraces(end+1, :) = vfStates(vnTraceIndices);

figure(hMonitorFigure);
clf;
plot(vtTime, mfStateTraces, 'c-');
hold on;
plot(vtTime, mfActivityTraces, '-');
