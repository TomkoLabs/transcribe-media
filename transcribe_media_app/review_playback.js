/* Listening context never changes the interval assigned or used for training. */
(function(root) {
  'use strict';
  function padding(mode, uncertain) {
    if(mode === 'exact') return 0;
    if(mode === 'wide') return 5;
    return uncertain ? 4 : 2;
  }
  function bounds(start, end, duration, context) {
    if(![start,end,duration,context].every(Number.isFinite) || start < 0 ||
       end <= start || duration <= 0 || start >= duration || context < 0)
      throw Error('This selection has invalid timestamps or lies outside the audio.');
    return {start:Math.max(0,start-context), end:Math.min(duration,end+context),
      targetStart:start, targetEnd:Math.min(duration,end)};
  }
  function phase(at, range) {
    if(at < range.targetStart) return 'Before target · context only';
    if(at < range.targetEnd) return 'Target · this is the selected speech';
    return 'After target · context only';
  }
  function mark(at, turn, other, edge) {
    if(!Number.isFinite(at) || at < turn.start || at > turn.end)
      throw Error('The playhead is outside this soundbite. Listen to it first, then mark a point inside its target times.');
    const value=Math.max(turn.start,Math.min(turn.end,Math.round(at*1000)/1000));
    if(!Number.isFinite(other) || (edge==='start' ? value>=other : value<=other))
      throw Error('Start must be before end. Adjust the other boundary first.');
    return value;
  }
  root.ReviewPlayback={padding,bounds,phase,mark};
  if(typeof module!=='undefined') module.exports=root.ReviewPlayback;
})(globalThis);
