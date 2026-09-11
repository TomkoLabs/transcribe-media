/* Shared, model-free review draft logic. Embedded into each offline review page. */
(function(root) {
  'use strict';
  const copy = value => JSON.parse(JSON.stringify(value));
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  class ReviewDraft {
    constructor(data) {
      this.data = data;
      this.locals = [...new Set(data.turns.map(t => t.local_speaker || t.speaker))];
      this.turns = new Map(data.turns.map(t => [t.id, t]));
      this.catalog = new Map(data.profiles.map(p => [p.voice_id, p]));
      this.defaults = {};
      for (const [label, role] of [['Adult A','adult'],['Adult B','adult'],['Child','child']]) {
        if (!data.profiles.some(p => p.label === label)) this.defaults['new:'+label] = {label,role};
      }
      this.machine = Object.fromEntries(data.matches.filter(m => m.status === 'matched').map(m => [m.local_speaker,m.speaker]));
      this.load({format_version:2, review_id:data.review_id, packet:data.packet, ...(data.applied_decisions || {})});
    }
    load(input) {
      if (!object(input) || input.review_id !== this.data.review_id || input.packet !== this.data.packet)
        throw Error('This file belongs to a different or outdated recording review. Open the matching review page.');
      if (input.format_version != null && input.format_version !== 2) throw Error('Unsupported review file version.');
      const draft = copy(input);
      for (const key of ['assignments','turn_overrides','new_profiles','profile_updates']) {
        if (!object(draft[key] || {})) throw Error(key+' must be an object.');
        draft[key] = draft[key] || {};
      }
      draft.range_overrides = draft.range_overrides || [];
      draft.exclude_windows = draft.exclude_windows || [];
      if (!Array.isArray(draft.range_overrides) || !Array.isArray(draft.exclude_windows)) throw Error('Invalid clip/range list.');
      const resolve = choice => (this.data.new_profile_ids || {})[choice] || choice;
      const validMetadata = p => object(p) && typeof p.label === 'string' && p.label.trim() && p.label.length <= 80 &&
        !/[\x00-\x1f]/.test(p.label) && ['adult','child','unspecified'].includes(p.role);
      for (const [key,p] of Object.entries(draft.new_profiles)) {
        if (!key.startsWith('new:') || !validMetadata(p)) throw Error('Invalid new profile label or role.');
      }
      for (const [key,p] of Object.entries(draft.profile_updates)) {
        if (!this.catalog.has(key) || !validMetadata(p) || (p.archived != null && typeof p.archived !== 'boolean'))
          throw Error('Invalid existing profile edit. Regenerate the review page if the catalog changed.');
      }
      const choices = Object.hasOwn(input,'new_profiles') ? draft.new_profiles : {...this.defaults};
      const check = choice => {
        choice = resolve(choice);
        if (typeof choice !== 'string' || !choice || (choice !== 'unknown' && !this.catalog.has(choice) && !choices[choice]))
          throw Error('Unknown profile choice. Regenerate the review page to load current profiles.');
        return choice;
      };
      for (const [local,choice] of Object.entries(draft.assignments)) {
        if (!this.locals.includes(local)) throw Error('A speaker in this file is absent from this recording.');
        draft.assignments[local] = check(choice);
      }
      for (const [id,choice] of Object.entries(draft.turn_overrides)) {
        if (!this.turns.has(id)) throw Error('A turn in this file is absent from this recording.');
        draft.turn_overrides[id] = check(choice);
      }
      for (const range of draft.range_overrides) {
        const turn = object(range) && this.turns.get(range.turn_id);
        if (!turn || !Number.isFinite(range.start) || !Number.isFinite(range.end) ||
            range.start < turn.start || range.end > turn.end || range.start >= range.end)
          throw Error('A correction must stay inside its turn and have an end after its start.');
        range.profile = check(range.profile);
      }
      for (const turn of this.data.turns) {
        const ranges = draft.range_overrides.filter(r => r.turn_id === turn.id).sort((a,b)=>a.start-b.start);
        if (ranges.some((r,i)=>i && r.start < ranges[i-1].end)) throw Error('Time corrections overlap.');
      }
      const windows = new Set(Object.entries(this.data.windows).flatMap(([local,list])=>list.map((w,i)=>local+':'+i)));
      if (draft.exclude_windows.some(id=>!windows.has(id))) throw Error('A reference clip is absent from this recording.');
      // Commit only after the complete import has validated.
      this.assignments = draft.assignments;
      this.overrides = draft.turn_overrides;
      this.ranges = draft.range_overrides;
      this.newProfiles = choices;
      this.updates = draft.profile_updates;
      this.excluded = new Set(draft.exclude_windows);
      if (!Object.hasOwn(input,'exclude_windows')) {
        for (const [local,list] of Object.entries(this.data.windows)) list.forEach((w,i)=>{
          if (!w.reference_eligible || w.retained === false || w.model_disagreement) this.excluded.add(local+':'+i);
        });
      }
    }
    groupChoice(local) { return this.assignments[local] || this.machine[local] || ''; }
    choiceAt(turn, at) {
      return this.ranges.find(r=>r.turn_id===turn.id && r.start <= at && at < r.end)?.profile ||
        this.overrides[turn.id] || this.groupChoice(turn.local_speaker || turn.speaker);
    }
    clipChoices(local,start,end) {
      const values = new Set();
      for (const turn of this.data.turns.filter(t=>(t.local_speaker||t.speaker)===local && t.start < end && t.end > start)) {
        const bounds = [...new Set([Math.max(start,turn.start),Math.min(end,turn.end),
          ...this.ranges.filter(r=>r.turn_id===turn.id).flatMap(r=>[r.start,r.end]).filter(t=>t>Math.max(start,turn.start)&&t<Math.min(end,turn.end))])].sort((a,b)=>a-b);
        for(let i=1;i<bounds.length;i++) values.add(this.choiceAt(turn,(bounds[i-1]+bounds[i])/2));
      }
      return [...values];
    }
    setRange(id,start,end,profile) {
      const next = this.export();
      next.range_overrides = this.ranges.flatMap(r=>{
        if(r.turn_id!==id || r.end<=start || r.start>=end) return [r];
        return [...(r.start<start?[{...r,end:start}]:[]),...(r.end>end?[{...r,start:end}]:[])];
      });
      if (profile) next.range_overrides.push({turn_id:id,start,end,profile});
      this.load(next);
    }
    setClip(local,start,end,profile) {
      const before = this.export();
      try {
        for (const turn of this.data.turns.filter(t=>(t.local_speaker||t.speaker)===local && t.start<end && t.end>start))
          this.setRange(turn.id,Math.max(start,turn.start),Math.min(end,turn.end),profile);
      } catch(error) { this.load(before); throw error; }
    }
    removeDraft(key) {
      delete this.newProfiles[key];
      for (const [id,value] of Object.entries(this.assignments)) if(value===key) this.assignments[id]='unknown';
      for (const [id,value] of Object.entries(this.overrides)) if(value===key) this.overrides[id]='unknown';
      this.ranges = this.ranges.map(r=>r.profile===key?{...r,profile:'unknown'}:r);
    }
    label(choice) {
      if(!choice) return 'Not assigned';
      if(choice==='unknown') return 'Unknown / review later';
      return this.newProfiles[choice]?.label || this.updates[choice]?.label || this.catalog.get(choice)?.label || choice;
    }
    status(local) {
      const turns=this.data.turns.filter(t=>(t.local_speaker||t.speaker)===local);
      const values=turns.flatMap(t=>this.clipChoices(local,t.start,t.end));
      if(values.some(v=>!v||v==='unknown')) return 'pending';
      if(this.assignments[local] || turns.some(t=>this.overrides[t.id] || this.ranges.some(r=>r.turn_id===t.id))) return 'reviewed';
      return this.machine[local] ? 'matched' : 'pending';
    }
    export() {
      return {format_version:2, review_id:this.data.review_id, packet:this.data.packet,
        assignments:copy(this.assignments),turn_overrides:copy(this.overrides),range_overrides:copy(this.ranges),
        new_profiles:copy(this.newProfiles),profile_updates:copy(this.updates),exclude_windows:[...this.excluded].sort()};
    }
  }
  root.ReviewDraft=ReviewDraft;
  if(typeof module!=='undefined') module.exports={ReviewDraft};
})(globalThis);
