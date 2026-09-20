'use strict';
const byId=id=>document.getElementById(id);
const inputData=JSON.parse(byId('review-data').textContent);
let workspace=new ReviewWorkspace(inputData),data=workspace.records[0],draft=workspace.drafts[0];
let filter='pending',stopAt=null,activeClip=null,shownLocals=null,projectHandle=null;
let playbackRange=null,playbackTitle='',playRequest=0,playTimer=null,lastQueueTurn=null;
const openDetails=new Set();
const storageKey='transcribe-review:'+(workspace.batch?inputData.batch_id:data.review_id)+':'+workspace.records.map(r=>r.draft_revision).join(':');
const filename=workspace.batch?'batch-'+inputData.batch_id.slice(0,16)+'.decisions.json':data.packet.replace(/\.json$/,'.decisions.json');
const command=ReviewIO.command(filename);
byId('command').textContent=command;
function selectRecording(index){stopPlayback();workspace.current=index;data=workspace.records[index];draft=workspace.drafts[index];
  byId('audio').src=data.packet.replace(/\.json$/,'.wav');byId('title').textContent=data.source;
  shownLocals=null;openDetails.clear();redraw();}
function message(text,error=false){byId('message').textContent=text;byId('message').classList.toggle('error',error)}
try{const saved=localStorage.getItem(storageKey);if(saved){workspace.load(JSON.parse(saved));draft=workspace.drafts[workspace.current];message('Restored your local draft. Changes are not applied to the transcript yet.')}else message('Review only what needs attention. Existing assignments are shown below.')}catch{message('Browser draft storage is unavailable or stale. Export a file to save your review.')}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(workspace.export()));message('Draft saved in this browser · export and apply to update the transcript.')}catch{message('Export your review to save it; browser storage is unavailable.')}}
function element(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e}
function time(value){let ms=Math.round(value*1000),hours=Math.floor(ms/3600000),minutes=Math.floor(ms/60000)%60,seconds=Math.floor(ms/1000)%60;return [hours,minutes,seconds].map(v=>String(v).padStart(2,'0')).join(':')+'.'+String(ms%1000).padStart(3,'0')}
function stopPlayback() {
  playRequest++;
  byId('audio').pause();
  clearInterval(playTimer);playTimer=null;stopAt=null;playbackRange=null;
  activeClip?.classList.remove('playing','target-playing');activeClip=null;
  byId('playback-info').textContent='Listen adds context; only the selected target is assigned or used for training.';
}
function audioReady(audio) {
  if(audio.readyState>=1 && Number.isFinite(audio.duration))return Promise.resolve();
  return new Promise((resolve,reject)=>{
    const finish=error=>{clearTimeout(timer);audio.removeEventListener('loadedmetadata',ready);audio.removeEventListener('error',failed);error?reject(error):resolve();};
    const ready=()=>finish(),failed=()=>finish(Error('Audio unavailable. Keep all review WAV files beside this HTML.'));
    const timer=setTimeout(()=>finish(Error('Audio did not load. Check the sibling WAV, then try Listen again.')),10000);
    audio.addEventListener('loadedmetadata',ready,{once:true});audio.addEventListener('error',failed,{once:true});
  });
}
function updatePlayback() {
  const audio=byId('audio');
  byId('playback-clock').textContent=time(audio.currentTime);
  if(!playbackRange)return;
  const phase=ReviewPlayback.phase(audio.currentTime,playbackRange);
  const text=playbackTitle+' · '+phase+' · target '+time(playbackRange.targetStart)+' → '+time(playbackRange.targetEnd)+
    ' · listening '+time(playbackRange.start)+' → '+time(playbackRange.end);
  if(byId('playback-info').textContent!==text)byId('playback-info').textContent=text;
  for(const word of activeClip?.querySelectorAll('.word')||[])word.classList.toggle('word-playing',audio.currentTime>=Number(word.dataset.start)&&audio.currentTime<Number(word.dataset.end));
  activeClip?.classList.toggle('target-playing',audio.currentTime>=playbackRange.targetStart&&audio.currentTime<playbackRange.targetEnd);
  if(stopAt!==null&&audio.currentTime>=stopAt){audio.pause();stopAt=null;}
}
async function listen(start,end,container,uncertain,title,sourceRange=null) {
  stopPlayback();
  const request=playRequest,audio=byId('audio');
  try {
    await audioReady(audio);
    if(request!==playRequest)return;
    playbackRange=ReviewPlayback.bounds(start,end,audio.duration,ReviewPlayback.padding(byId('context').value,uncertain));
    if(sourceRange){const source=ReviewPlayback.bounds(sourceRange.start,sourceRange.end,audio.duration,0);
      playbackRange={...playbackRange,start:source.start,end:source.end};}
    playbackTitle=title;activeClip=container;container.classList.add('playing');
    stopAt=playbackRange.end;audio.currentTime=playbackRange.start;
    updatePlayback();await audio.play();
  } catch(error) {
    if(request!==playRequest)return;
    stopPlayback();message(error.message+' If playback is blocked, press Play in the audio player once and retry.',true);
  }
}
function playButton(start,end,container,uncertain=false,title='Selection') {
  const button=element('button','▶ Listen','small');button.type='button';
  button.setAttribute('aria-label','Listen '+time(start)+' to '+time(end));
  button.onclick=()=>listen(start,end,container,uncertain,title);return button;
}
byId('audio').addEventListener('timeupdate',updatePlayback);
byId('audio').addEventListener('play',()=>{
  if(playbackRange&&byId('audio').currentTime>=playbackRange.end){playbackRange=null;stopAt=null;activeClip?.classList.remove('playing','target-playing');activeClip=null;byId('playback-info').textContent='Listening freely. The playback clock can be used to mark a correction.';}
  activeClip?.classList.add('playing');clearInterval(playTimer);playTimer=setInterval(updatePlayback,50);
});
byId('audio').addEventListener('pause',()=>{clearInterval(playTimer);playTimer=null;activeClip?.classList.remove('playing','target-playing');});
byId('context').onchange=()=>stopPlayback();
function profileSelect(value,label,blank){const select=document.createElement('select');select.setAttribute('aria-label',label);select.add(new Option(blank||'Not assigned — choose a person',''));select.add(new Option('Unknown / review later','unknown'));select.add(new Option('UNKNOWN — exclude from voice learning','ignore'));
for(const p of data.profiles){const current={...p,...draft.updates[p.voice_id]};select.add(new Option(current.label+' · '+p.voice_id+(current.archived?' (archived)':''),p.voice_id))}
for(const [key,p] of Object.entries(draft.newProfiles)){if(!(draft.data.new_profile_ids||{})[key])select.add(new Option('New: '+p.label,key))}select.value=value||'';return select}
function turnAttention(turn){return !!turn.review_reasons?.length||draft.turnRisk(turn)}
function attention(local){return draft.needsAttention(local)}
function updateCounts(){
const saved=data.saved_status;
if(saved)byId('saved-status').textContent='Saved state: '+saved.state+' · '+(saved.pending??'unknown')+' pending item(s) · Decisions '+(data.decisions_allowed?'allowed':'BLOCKED')+'. '+saved.action+' Snapshot '+saved.snapshot_id+' · generated '+saved.generated_utc+' · version '+(saved.program_version||inputData.program_version||'unknown')+'. Offline snapshot: regenerate and reopen after CLI changes.';
const canExport=workspace.records.some(r=>r.decisions_allowed!==false);
for(const id of ['save','save-project','copy'])byId(id).disabled=!canExport;
const pending=draft.locals.filter(attention).length,assigned=draft.locals.length-pending;byId('progress').textContent=data.decisions_allowed===false?'Blocked saved source. Follow the recovery action above; browser edits cannot finalize it.':'Unapplied browser draft: '+draft.pendingTurns().length+' soundbite'+(draft.pendingTurns().length===1?'':'s')+' needing attention · '+pending+' voice group'+(pending===1?'':'s')+' needing attention · '+assigned+' already assigned';byId('filter-pending').textContent='Needs attention ('+pending+')';byId('filter-assigned').textContent='Already assigned ('+assigned+')';byId('profile-count').textContent=data.profiles.length+' existing profiles · '+data.profiles.reduce((n,p)=>n+(p.verified_windows||0),0)+' verified reference clips retained across recordings';for(const name of ['pending','assigned','all'])byId('filter-'+name).setAttribute('aria-pressed',String(filter===name))}
function redraw(){const y=window.scrollY;renderPeople();renderCards();updateCounts();updateNavigation();window.scrollTo(0,y)}
function changed(){stopPlayback();workspace.synchronize(draft);persist();redraw()}
function renderPeople(){const list=byId('profile-list');list.replaceChildren();
for(const p of data.profiles){const current={...p,...draft.updates[p.voice_id]},row=element('div',null,'person');row.append(element('strong',p.voice_id+' · '+(current.archived?'Archived':p.training_status==='ready'?'Ready · reference support available':p.verified_windows?'Collecting · more voice references needed':'Untrained · transcript label saved')));row.append(element('p',(p.verified_seconds||0)+'s verified · '+p.verified_windows+' clip'+(p.verified_windows===1?'':'s')+' · '+p.verified_sessions+' recording'+(p.verified_sessions===1?'':'s'),'muted'));
const form=element('div',null,'row'),label=element('label','Label'),input=element('input');input.type='text';input.maxLength=80;input.value=current.label;input.setAttribute('aria-label','Label for '+p.voice_id);label.append(input);
const roleLabel=element('label','Role'),role=element('select');role.setAttribute('aria-label','Role for '+p.voice_id);for(const value of ['unspecified','adult','child'])role.add(new Option(value,value));role.value=current.role;roleLabel.append(role);
const archived=element('input');archived.type='checkbox';archived.checked=!!current.archived;const archiveLabel=element('label',null,'checkbox');archiveLabel.append(archived,document.createTextNode(' Archive for future matching'));
const save=element('button','Save profile edit');save.onclick=()=>{const value=input.value.trim();if(!value||/[\x00-\x1f]/.test(value))return message('Enter a valid profile label.',true);draft.updates[p.voice_id]={label:value,role:role.value,archived:archived.checked};changed()};form.append(label,roleLabel,archiveLabel,save);row.append(form);list.append(row)}
for(const [key,p] of Object.entries(draft.newProfiles)){if((draft.data.new_profile_ids||{})[key])continue;const row=element('div',null,'person row');const label=element('input');label.type='text';label.maxLength=80;label.value=p.label;label.setAttribute('aria-label','Draft profile '+p.label);const role=element('select');role.setAttribute('aria-label','Role for draft '+p.label);for(const value of ['unspecified','adult','child'])role.add(new Option(value,value));role.value=p.role;const save=element('button','Update label');save.onclick=()=>{if(!label.value.trim()||/[\x00-\x1f]/.test(label.value))return message('Enter a valid label.',true);draft.newProfiles[key]={label:label.value.trim(),role:role.value};changed()};const remove=element('button','Remove draft person');remove.onclick=()=>{workspace.removePerson(key);changed()};row.append(element('span','New person'),label,role,save,remove);list.append(row)}}
function renderCards(){for(const details of document.querySelectorAll('details[data-open-key]')){details.open?openDetails.add(details.dataset.openKey):openDetails.delete(details.dataset.openKey)}
if(shownLocals===null)shownLocals=new Set(draft.locals.filter(local=>filter==='all'||(filter==='pending'?attention(local):!attention(local))));
byId('cards').replaceChildren();
if(data.decisions_allowed===false){byId('cards').append(element('p','This saved source is blocked. Recover it before reviewing or exporting its decisions.','warning'));return;}
let visible=0;
for(const local of draft.locals){const needs=attention(local),status=draft.status(local);if(!shownLocals.has(local))continue;visible++;
const turns=data.turns.filter(t=>(t.local_speaker||t.speaker)===local),match=data.matches.find(m=>m.local_speaker===local),section=element('section',null,'group'+(needs?' needs-review':'')),head=element('div',null,'group-head'),heading=element('div',null,'group-heading');heading.append(element('h2',local+' · '+turns.length+' soundbite'+(turns.length===1?'':'s')),element('span',status==='matched'?(needs?'Check speaker timing':'Confident automatic match'):status==='reviewed'?(draft.machine[local]&&!draft.assignments[local]?'Automatic match + your corrections':'Chosen by you'):(draft.groupChoice(local)?'Review exceptions':'Choose a person'),'badge '+status));head.append(heading);
const candidates=match?.reference_candidates?.length?match.reference_candidates:match?.candidates;
if(candidates?.length){const text=candidates.map(c=>draft.label(c.speaker)+' '+c.similarity.toFixed(3)).join(' · ');head.append(element('p',(match.reference_candidates?.length?'Verified reference matches (similarity): ':'Overall voice resemblance (suggestions only): ')+text,'muted'))}else head.append(element('p',status==='pending'?'No trained match yet. Reuse an existing person or choose a new label.':'Your choices below identify the speech in this group.','muted'));
const diagnostic=match?.matching_diagnostics;
if(diagnostic){
  if(Number.isFinite(diagnostic.compared_windows))head.append(element('p',diagnostic.supporting_windows+' / '+diagnostic.compared_windows+' comparable voice samples support this match; '+diagnostic.model_disputed_windows+' disputed samples handled separately.','muted'));
  if(status==='pending'&&diagnostic.blockers?.length)head.append(element('p','Why this group is unassigned: '+diagnostic.blockers.map(reason=>({
    insufficient_query_audio:'too little usable speech',no_verified_profiles:'no trained reference profile yet',
    insufficient_independent_windows:'too few independent speech samples',inconsistent_window_matches:'voice samples do not consistently agree',
    unresolved_disputed_voice_evidence:'disputed samples do not confirm the same known voice',
    mixed_voice_evidence:'acoustic samples suggest multiple voices',insufficient_verified_reference_match:'verified-reference similarity is below the required threshold',
    ambiguous_profile_match:'the best profiles are too close',overlap_with_same_profile:'two simultaneous groups cannot be assigned to the same person'
  })[reason]||reason.replaceAll('_',' ')).join('; ')+'.','warning'));
}
const referenceProgress=draft.referenceProgress(local);
if(referenceProgress.total)head.append(element('p',referenceProgress.reviewed+' / '+referenceProgress.total+' reference clips approved from this recording (earlier verified references are retained)','muted'));
if(status==='pending'&&referenceProgress.total&&referenceProgress.reviewed===referenceProgress.total){
  head.append(element('p','Reference clips are assigned; other spoken words still need a person. Assign the remaining unflagged speech after listening; flagged exceptions still need individual choices.','warning'));
  if(referenceProgress.person&&referenceProgress.person!=='ignore'){
    const complete=element('button','Assign remaining speech to '+draft.label(referenceProgress.person),'small');
    complete.onclick=()=>{draft.assignGroup(local,referenceProgress.person);changed()};head.append(complete);
  }
}
if(data.mixed.includes(local))head.append(element('p','This group may contain different people. Use clip or turn corrections for the exceptions.','warning'));
const assign=element('div',null,'assignment'),label=element('label','Default person for this group'),id='group-'+draft.locals.indexOf(local);label.htmlFor=id;const select=profileSelect(draft.groupChoice(local),'Assign '+local,status==='reviewed'?'Individual soundbites assigned below':null);select.id=id;select.dataset.local=local;select.onchange=()=>{draft.assignGroup(local,select.value);changed()};assign.append(label,select);head.append(assign,element('p','Applies to unflagged speech. Flagged soundbites stay for individual review; existing corrections are preserved.','muted'));if(draft.machine[local]&&!draft.assignments[local]){const confirm=element('button','Confirm this person & learn','small');confirm.onclick=()=>{draft.assignGroup(local,draft.groupChoice(local));changed()};head.append(element('p','Automatic matches do not train themselves. After listening, optionally confirm this person to approve the selected reference clips.','muted'),confirm)}section.append(head);
const body=element('div',null,'group-body'),windows=data.windows[local]||[],refs=element('details');refs.dataset.openKey='training:'+local;refs.open=openDetails.has(refs.dataset.openKey);refs.append(element('summary','Voice learning reference clips · '+windows.length+' sample'+(windows.length===1?'':'s')),element('p','These are acoustic samples, not whole transcript sentences. Train only clips you have listened to.','muted'));
const samples=element('div',null,'samples'),moreSamples=element('div',null,'samples'),more=element('details');more.dataset.openKey='references:'+local;more.open=openDetails.has(more.dataset.openKey);more.append(element('summary','Show all '+windows.length+' reference clips'),moreSamples);const firstClips=new Set(windows.length<=6?windows.map((_,i)=>i):Array.from({length:6},(_,i)=>Math.round(i*(windows.length-1)/5)));windows.forEach((w,i)=>{const values=draft.clipChoices(local,w.start,w.end),assigned=values.length===1?draft.label(values[0]):'Mixed assignments — inspect turns',clip=element('article',null,'clip'+(w.retained===false||w.model_disagreement?' attention':''));clip.dataset.clip=local+':'+i;const title=element('div',null,'clip-head');title.append(element('strong','Clip '+(i+1)),playButton(w.start,w.end,clip,w.model_disagreement||w.retained===false||turns.some(t=>t.start<w.end&&t.end>w.start&&turnAttention(t)),local+' · clip '+(i+1)));clip.append(title,element('div',time(w.start)+' → '+time(w.end)+' · '+w.duration.toFixed(1)+'s','timestamp'));
const excerpt=ReviewPlayback.clipText(turns,w.start,w.end);
clip.append(element('p',excerpt.text||'No complete aligned words inside this reference clip. Use the soundbites above for transcript review.','excerpt'));
if(excerpt.partial)clip.append(element('p','Some words cross the clip boundary and are omitted from this excerpt.','muted'));
clip.append(element('div','Speaker: '+assigned,'assigned'));
if(w.candidates?.length)clip.append(element('p','Reference comparison: '+w.candidates.map(c=>draft.label(c.speaker)+' '+c.similarity.toFixed(3)).join(' · ')+' (similarity, not probability)','muted'));
const correction=profileSelect('', 'Correct speaker for '+local+' clip '+(i+1),'Change only this clip…');correction.onchange=()=>{if(correction.value){draft.setClip(local,w.start,w.end,correction.value);changed()}};clip.append(correction);
const excludedIdentity=values.some(value=>value==='unknown'||value==='ignore');const check=element('input');check.type='checkbox';check.dataset.window=local+':'+i;check.checked=!excludedIdentity&&!draft.excluded.has(local+':'+i)&&!!w.reference_eligible;check.disabled=excludedIdentity||!w.reference_eligible;check.onchange=()=>{check.checked?draft.excluded.delete(local+':'+i):draft.excluded.add(local+':'+i);persist()};const ref=element('label',null,'reference'+(!w.reference_eligible?' disabled':''));ref.append(check,document.createTextNode(excludedIdentity?'Excluded from voice learning: unknown speaker':w.reference_eligible?'Use this clip to learn the assigned voice':'Transcript correction only; unsuitable training audio'));clip.append(ref);
if(w.retained===false||w.model_disagreement)clip.append(element('p','Extra verification: models or clips disagree. Listen before including this reference.','muted'));
else if(w.duration<2.5)clip.append(element('p','Short reference: contributes with other consistent reviewed clips.','muted'));(firstClips.has(i)?samples:moreSamples).append(clip)});refs.append(samples);if(windows.length>6)refs.append(more);
if(!windows.length)refs.append(element('p','No suitable reference clips were extracted. You can still assign a person and save the transcript. Their voice can learn from later recordings.','notice'));
const details=element('details');details.dataset.openKey='turns:'+local;details.open=openDetails.has(details.dataset.openKey)||turns.some(t=>draft.turnNeedsAttention(t));details.append(element('summary',(filter==='pending'?'Needs attention: '+turns.filter(t=>draft.turnNeedsAttention(t)).length+' of ': 'All ')+turns.length+' soundbite'+(turns.length===1?'':'s')+' · inspect or correct individual turns'));
for(const [index,t] of turns.entries()){if(filter==='pending'&&!draft.turnNeedsAttention(t))continue;const row=element('article',null,'turn'+(turnAttention(t)?' attention':''));row.dataset.turnCard=t.id;const top=element('div',null,'clip-head');top.append(element('strong','Soundbite '+(index+1)),playButton(t.start,t.end,row,turnAttention(t),local+' · soundbite '+(index+1)));row.append(top,element('div',time(t.start)+' → '+time(t.end),'timestamp'),element('p',t.text));row.append(element('span',draft.turnNeedsAttention(t)?'Needs your review':draft.spokenChoices(t,true).every(v=>v&&v!=='unknown')?'Chosen by you':'Confident automatic match','badge '+(draft.turnNeedsAttention(t)?'pending':'reviewed')));const values=draft.spokenChoices(t);row.append(element('div','Speaker: '+(values.length===1?draft.label(values[0]):'Mixed — see time corrections'),'assigned'));if(turnAttention(t))row.append(element('p',(draft.turnRisk(t)?'Speaker check: ':'Timing/text note — speaker choice is retained: ')+(t.review_reasons||['speaker timing disputed']).map(r=>r.replace(/^check_/, '').replaceAll('_',' ')).join(' · ')+'. If words are missing from playback, try the source sentence below.','warning'));
if(t.suggested_speakers?.length)row.append(element('p','Voice-reference suggestions: '+t.suggested_speakers.map(id=>draft.label(id)).join(' / ')+'. Listen before selecting.','muted'));
else if(draft.machine[local]&&draft.turnNeedsAttention(t))row.append(element('p','Group match suggests '+draft.label(draft.machine[local])+', but this soundbite still needs checking.','muted'));
const source=t.review_timing?.source_context;
if(source){const sentence=playButton(source.start,source.end,row,false,'Original ASR sentence');sentence.textContent='▶ Listen to source sentence';sentence.onclick=()=>listen(t.start,t.end,row,false,'Source sentence · correction target unchanged',source);sentence.setAttribute('aria-label','Listen to source sentence for '+t.id);row.append(sentence,element('p',source.text,'source-excerpt'));}
const wordList=element('div',null,'word-list');
for(const word of t.words||[]){if(!Number.isFinite(word.start)||!Number.isFinite(word.end)||word.end<=word.start)continue;const button=playButton(word.start,word.end,row,!!word.timing?.issues?.length,'Word: '+(word.text||word.word));button.textContent=word.text||word.word;button.classList.add('word');button.dataset.start=word.start;button.dataset.end=word.end;button.title=time(word.start)+' → '+time(word.end)+(word.timing?.issues?.length?' · timing uncertain':'');wordList.append(button);}
if(wordList.childElementCount)row.append(element('p','Click a word to hear its aligned position. The source sentence uses the recognizer’s original times.','muted'),wordList);
const choice=profileSelect(draft.overrides[t.id]||'','Speaker for '+t.id,'Use group assignment');choice.dataset.turn=t.id;choice.onchange=()=>{if(choice.value)draft.overrides[t.id]=choice.value;else delete draft.overrides[t.id];draft.ranges=draft.ranges.filter(r=>r.turn_id!==t.id);changed()};row.append(choice);
const ranges=element('details');ranges.dataset.openKey='ranges:'+t.id;ranges.open=openDetails.has(ranges.dataset.openKey)||draft.ranges.some(r=>r.turn_id===t.id);ranges.append(element('summary','Correct part of this soundbite'));
for(const r of draft.ranges.filter(r=>r.turn_id===t.id)){const item=element('div',null,'range'),remove=element('button','Remove correction','small');remove.onclick=()=>{draft.ranges=draft.ranges.filter(item=>item!==r);changed()};item.append(element('span',time(r.start)+' → '+time(r.end)+' · '+draft.label(r.profile)),playButton(r.start,r.end,row,turnAttention(t),'Saved correction'),remove);ranges.append(item)}
const controls=element('div',null,'row'),start=element('input'),end=element('input');
for(const [input,key] of [[start,'start'],[end,'end']]) {
  input.type='number';input.step='.001';input.min=t.start;input.max=t.end;input.value=t[key];
  input.setAttribute('aria-label',key+' seconds for '+t.id);
  const label=element('label',key+' seconds ');label.append(input);controls.append(label);
}
const preview=element('button','Preview selection'),markStart=element('button','Start at playhead'),markEnd=element('button','End at playhead');
function selection() {
  const first=start.valueAsNumber,last=end.valueAsNumber;
  if(!Number.isFinite(first)||!Number.isFinite(last)||first<t.start||last>t.end||first>=last)
    throw Error('Choose a start before the end, within this soundbite’s target times.');
  return [first,last];
}
preview.onclick=()=>{try{const [first,last]=selection();listen(first,last,row,turnAttention(t),'Correction preview')}catch(error){message(error.message,true)}};
for(const [button,input,other,edge] of [[markStart,start,end,'start'],[markEnd,end,start,'end']]) {
  button.onclick=()=>{try{input.value=ReviewPlayback.mark(byId('audio').currentTime,t,other.valueAsNumber,edge);
    message('Marked '+edge+' at '+time(Number(input.value))+'. Preview the selection, choose a person, then Set time correction.');
  }catch(error){message(error.message,true)}};
}
const rangeChoice=profileSelect('','Person for a partial correction in '+t.id),apply=element('button','Set time correction');
apply.onclick=()=>{try{const [first,last]=selection();if(!rangeChoice.value)throw Error('Choose a person for this time correction.');draft.setRange(t.id,first,last,rangeChoice.value);changed()}catch(error){message(error.message,true)}};
const audition=element('div',null,'row');audition.append(preview,markStart,markEnd);
controls.append(rangeChoice,apply);
ranges.append(element('p','Listen, pause near a speaker change, and mark the playhead or type seconds. Preview before applying.','muted'),
  audition,controls,element('p','Context is for listening only. Corrections use word midpoints inside the selected start/end; recognized words and training clip boundaries stay unchanged.','muted'));
row.append(ranges);details.append(row)}body.append(details,refs);section.append(body);byId('cards').append(section)}
if(!visible)byId('cards').append(element('div',filter==='pending'?'No unassigned groups in this recording. Move to the next recording, save changes, or open Already assigned for a spot check.':'No voice groups in this view.','empty'))}
byId('add').onclick=()=>{const label=byId('name').value.trim();if(!label||/[\x00-\x1f]/.test(label))return message('Enter a valid person label.',true);if([...data.profiles,...Object.values(draft.newProfiles)].some(p=>p.label.toLowerCase()===label.toLowerCase()))return message('That label already exists. Choose that person instead of creating a duplicate.',true);draft.newProfiles[workspace.newKey()]={label,role:byId('role').value};byId('name').value='';changed()};
for(const name of ['pending','assigned','all'])byId('filter-'+name).onclick=()=>{filter=name;shownLocals=null;renderCards();updateCounts()};
byId('import').onclick=()=>byId('file').click();
byId('file').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;if(file.size>20000000)throw Error('Review file is too large. Choose an exported decisions JSON.');workspace.load(JSON.parse(await file.text()));draft=workspace.drafts[workspace.current];shownLocals=null;changed();message('Saved review imported. Nothing has been applied to the transcript yet.')}catch(error){message('Import failed: '+error.message,true)}finally{event.target.value=''}};
byId('save').onclick=()=>{try{const output=workspace.export();workspace.load(output);draft=workspace.drafts[workspace.current];const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);byId('apply-help').open=true;message('Download requested. Save or move the JSON to '+ReviewIO.relativePath(filename)+' inside your project BEFORE running the apply command.')}catch(error){message(error.message,true)}};
byId('copy').onclick=async()=>{byId('apply-help').open=true;try{await navigator.clipboard.writeText(command);message('Apply command copied. Ensure the JSON is in '+ReviewIO.relativePath(filename)+', then run from your project folder.')}catch{message('Select and copy the apply command shown below.')}};
byId('reset').onclick=()=>{workspace=new ReviewWorkspace(inputData);draft=workspace.drafts[0];data=workspace.records[0];shownLocals=null;try{localStorage.removeItem(storageKey)}catch{}selectRecording(0);message('Draft reset to the last applied review. Transcript files have not changed.')};
function updateNavigation(){
  byId('recording-nav').hidden=!workspace.batch;
  if(!workspace.batch)return;
  const select=byId('recording');select.replaceChildren();
  const counts=workspace.drafts.map(item=>item.pendingTurns().length);
  workspace.records.forEach((record,i)=>select.add(new Option((i+1)+'. '+record.source+' · '+(record.saved_status?record.saved_status.state+' · '+(record.saved_status.pending??'unknown')+' saved pending':counts[i]+' draft need attention'),String(i))));
  select.value=String(workspace.current);
  byId('batch-progress').textContent=workspace.records.length+' recordings · '+workspace.records.filter(r=>r.decisions_allowed===false).length+' blocked (excluded from export) · '+counts.reduce((a,b)=>a+b,0)+' soundbites need attention in unapplied browser drafts';
}
byId('recording').onchange=()=>selectRecording(Number(byId('recording').value));
byId('next-pending').onclick=()=>{for(let offset=1;offset<=workspace.records.length;offset++){
  const index=(workspace.current+offset)%workspace.records.length;
  if(workspace.drafts[index].locals.some(local=>workspace.drafts[index].needsAttention(local))){selectRecording(index);return;}}
  message('No more browser draft soundbites. Check Saved state; blocked recordings require recovery. Export eligible reviews and apply to update transcripts.');};
const folderSupported=ReviewIO.folderSupport(window);
byId('save-project').hidden=!folderSupported;
byId('folder-status').textContent=folderSupported?
  'Optional: your browser may let you select the project folder and save directly. Download JSON works without folder permission.':
  'Direct folder saving is unavailable in this browser. Download JSON, move it into speaker-decisions/, then run the command above.';
byId('save-project').onclick=async()=>{try{
  const output=workspace.export();workspace.load(output);draft=workspace.drafts[workspace.current];
  if(!projectHandle)projectHandle=await window.showDirectoryPicker({id:'transcribe-media-project',mode:'readwrite'});
  const saved=await ReviewIO.writeToProject(projectHandle,filename,output);
  byId('apply-help').open=true;message('Saved '+saved+' in '+projectHandle.name+'. Run the apply command from that project folder.');
}catch(error){
  projectHandle=null;const failure=ReviewIO.folderFailure(error);byId('apply-help').open=true;
  byId('folder-status').textContent=failure.text;if(failure.blocked)byId('save-project').hidden=true;
  message(failure.text,failure.blocked);
}};
selectRecording(0);

byId('next-soundbite').onclick=()=>{
  let pending=draft.pendingTurns();
  if(!pending.length){byId('next-pending').click();pending=draft.pendingTurns();}
  if(!pending.length)return message('No soundbites need attention. Download and apply your review.');
  filter='pending';shownLocals=null;redraw();
  const previous=pending.findIndex(t=>t.id===lastQueueTurn?.id&&workspace.current===lastQueueTurn.recording);
  const next=pending[(previous+1)%pending.length];lastQueueTurn={id:next.id,recording:workspace.current};
  const card=document.querySelector('[data-turn-card="'+next.id+'"]');
  card?.scrollIntoView({block:'center',behavior:'smooth'});
  card?.querySelector('button')?.focus({preventScroll:true});
};
