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
        if (typeof choice !== 'string' || !choice || (!['unknown','ignore'].includes(choice) && !this.catalog.has(choice) && !choices[choice]))
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
    turnRisk(turn) {
      if(turn.identity_review_reasons)return turn.identity_review_reasons.length>0;
      return turn.review_reasons ? turn.review_reasons.length>0 : turn.speaker_attribution?.status==='uncertain';
    }
    assignGroup(local, profile) {
      // Preserve explicit exceptions. A new blanket choice cannot approve a
      // disputed toddler/overlap/outlier interval without listening to it.
      for(const turn of this.data.turns.filter(t=>(t.local_speaker||t.speaker)===local)) {
        const chosen=this.spokenChoices(turn,true);
        if(profile && this.turnRisk(turn) && chosen.some(v=>!v||v==='unknown') && !this.overrides[turn.id])
          this.overrides[turn.id]='unknown';
      }
      if(profile)this.assignments[local]=profile;else delete this.assignments[local];
    }
    turnNeedsAttention(turn) {
      return this.spokenChoices(turn).some(choice=>!choice||choice==='unknown');
    }
    pendingTurns() { return this.data.turns.filter(t=>this.turnNeedsAttention(t)); }
    choiceAt(turn, at, humanOnly=false) {
      return this.ranges.find(r=>r.turn_id===turn.id && r.start <= at &&
        (at < r.end || at===turn.end && r.end===turn.end))?.profile ||
        this.overrides[turn.id] || this.assignments[turn.local_speaker||turn.speaker] ||
        (!humanOnly && !this.turnRisk(turn) ? this.machine[turn.local_speaker||turn.speaker]||'' : '');
    }
    spokenChoices(turn, humanOnly=false) {
      const units=turn.words?.length?turn.words:[turn];
      return [...new Set(units.map(unit=>Number.isFinite(unit.start)&&Number.isFinite(unit.end)?
        this.choiceAt(turn,(unit.start+unit.end)/2,humanOnly):this.overrides[turn.id]||''))];
    }
    clipChoices(local,start,end,humanOnly=false) {
      const values = new Set();
      for (const turn of this.data.turns.filter(t=>(t.local_speaker||t.speaker)===local && t.start < end && t.end > start)) {
        const bounds = [...new Set([Math.max(start,turn.start),Math.min(end,turn.end),
          ...this.ranges.filter(r=>r.turn_id===turn.id).flatMap(r=>[r.start,r.end]).filter(t=>t>Math.max(start,turn.start)&&t<Math.min(end,turn.end))])].sort((a,b)=>a-b);
        for(let i=1;i<bounds.length;i++) values.add(this.choiceAt(turn,(bounds[i-1]+bounds[i])/2,humanOnly));
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
      if(choice==='ignore') return 'UNKNOWN · excluded from voice learning';
      return this.newProfiles[choice]?.label || this.updates[choice]?.label || this.catalog.get(choice)?.label || choice;
    }
    status(local) {
      const turns=this.data.turns.filter(t=>(t.local_speaker||t.speaker)===local);
      const values=turns.flatMap(t=>this.spokenChoices(t));
      if(values.some(v=>!v||v==='unknown')) return 'pending';
      if(this.assignments[local] || turns.some(t=>this.overrides[t.id] || this.ranges.some(r=>r.turn_id===t.id))) return 'reviewed';
      return this.machine[local] ? 'matched' : 'pending';
    }
    needsAttention(local) {
      return this.data.turns.some(t=>(t.local_speaker||t.speaker)===local && this.turnNeedsAttention(t));
    }
    referenceProgress(local) {
      const clips=this.data.windows[local]||[];
      const reviewed=clips.filter(clip=>{
        const values=this.clipChoices(local,clip.start,clip.end,true);
        return values.length && values.every(choice=>choice&&choice!=='unknown');
      });
      const people=new Set(reviewed.flatMap(clip=>this.clipChoices(local,clip.start,clip.end,true)));
      return {reviewed:reviewed.length,total:clips.length,
        person:reviewed.length===clips.length&&clips.length&&people.size===1?[...people][0]:null};
    }
    export() {
      return {format_version:2, review_id:this.data.review_id, packet:this.data.packet,
        assignments:copy(this.assignments),turn_overrides:copy(this.overrides),range_overrides:copy(this.ranges),
        new_profiles:copy(this.newProfiles),profile_updates:copy(this.updates),exclude_windows:[...this.excluded].sort()};
    }
  }
  class ReviewWorkspace {
    constructor(input) {
      this.input=input;
      this.batch=input.kind==='speaker_review_batch';
      this.records=this.batch?input.recordings:[input];
      const aliases={...(input.new_profile_ids||{})},conflicts=new Set();
      for(const record of this.records)for(const [key,value] of Object.entries(record.new_profile_ids||{})){
        if(aliases[key]&&aliases[key]!==value)conflicts.add(key);else aliases[key]=value;
      }
      for(const key of conflicts)delete aliases[key];
      this.drafts=this.records.map(record=>new ReviewDraft({...record,
        new_profile_ids:{...aliases,...(record.new_profile_ids||{})}}));
      this.current=0;
      if(this.batch){
        const defaults={};
        for(const [label,role] of [['Adult A','adult'],['Adult B','adult'],['Child','child']]){
          if(!input.profiles.some(p=>p.label===label)) defaults['new:batch:'+input.batch_id.slice(0,12)+':'+label]={label,role};
        }
        this.drafts[0].newProfiles=defaults;
      }
      this.synchronize(this.drafts[0]);
    }
    synchronize(draft) {
      this.newProfiles=draft.newProfiles;this.updates=draft.updates;
      for(const item of this.drafts){item.newProfiles=this.newProfiles;item.updates=this.updates;}
    }
    removePerson(key){for(const draft of this.drafts)draft.removeDraft(key);this.synchronize(this.drafts[this.current]);}
    newKey(){let n=1;const prefix=this.batch?'new:batch:'+this.input.batch_id.slice(0,12)+':person-':'new:person-';
      while(this.newProfiles[prefix+n] || this.drafts.some(d=>(d.data.new_profile_ids||{})[prefix+n]))n++;return prefix+n;}
    export(){
      const allowed=this.drafts.filter(d=>d.data.decisions_allowed!==false);
      if(!allowed.length)throw Error('No current reviews can be exported. Follow the saved source recovery action.');
      if(!this.batch)return allowed[0].export();
      return {kind:'speaker_review_batch',format_version:3,batch_id:this.input.batch_id,
        new_profiles:copy(this.newProfiles),profile_updates:copy(this.updates),
        reviews:allowed.map(d=>({...d.export(),new_profiles:{},profile_updates:{}}))};
    }
    load(input){
      // Validate into temporary drafts so one bad recording cannot partially import.
      const isBatch=object(input)&&input.kind==='speaker_review_batch';
      if(isBatch && (input.format_version!==3 || !Array.isArray(input.reviews) || !input.reviews.length))throw Error('Invalid batch review file.');
      const entries=isBatch?input.reviews:[input];
      if(entries.some(e=>!object(e)) || new Set(entries.map(e=>e.packet)).size!==entries.length)throw Error('Invalid or duplicate recordings in review file.');
      if(entries.some(e=>this.records.find(r=>r.packet===e.packet)?.decisions_allowed===false))throw Error('This source is stale; recover it before importing decisions.');
      const incoming=new Map(entries.map(e=>[e.packet,e]));
      if([...incoming.keys()].some(key=>!this.records.some(r=>r.packet===key)))throw Error('This file includes recordings absent from this review page. Regenerate the index.');
      const complete=isBatch&&entries.length===this.records.length;
      const definitions=Object.hasOwn(input,'new_profiles')?input.new_profiles:{};
      const edits=Object.hasOwn(input,'profile_updates')?input.profile_updates:{};
      if(!object(definitions)||!object(edits))throw Error('Invalid shared profiles.');
      const newProfiles={...(complete?{}:this.newProfiles),...definitions};
      const updates={...(complete?{}:this.updates),...edits};
      const next=this.drafts.map(previous=>{
        const candidate=new ReviewDraft(previous.data);
        const entry=incoming.get(previous.data.packet)||previous.export();
        candidate.load({...entry,new_profiles:newProfiles,profile_updates:updates});return candidate;
      });
      this.drafts=next;this.synchronize(next[this.current]);
    }
  }
  root.ReviewDraft=ReviewDraft;
  root.ReviewWorkspace=ReviewWorkspace;
  if(typeof module!=='undefined') module.exports={ReviewDraft,ReviewWorkspace};
})(globalThis);
