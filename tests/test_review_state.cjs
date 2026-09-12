const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {ReviewDraft,ReviewWorkspace}=require('../transcribe_media_app/review_state.js');
const ReviewIO=require('../transcribe_media_app/review_io.js');
const make=()=>({review_id:'review-1',packet:'packet.json',source:'synthetic.wav',
  profiles:[{voice_id:'VOICE_0001',label:'Alex',role:'adult',verified_sessions:2}],
  matches:[{local_speaker:'A',speaker:'VOICE_0001',status:'matched',similarity:.92},
           {local_speaker:'B',speaker:'B',status:'unresolved_ambiguous'}],
  turns:[{id:'a',speaker:'A',start:0,end:6,text:'A voice'}, {id:'b',speaker:'B',start:7,end:12,text:'Another voice'}],
  windows:{A:[{start:0,end:6,duration:6,reference_eligible:true,retained:true}],
           B:[{start:7,end:9,duration:2,reference_eligible:true,retained:false}]},applied_decisions:{}});

test('confident assignments are preselected without being exported as human training decisions',()=>{
  const d=new ReviewDraft(make()); assert.equal(d.groupChoice('A'),'VOICE_0001');assert.equal(d.status('A'),'matched');
  assert.equal(d.groupChoice('B'),'');assert.equal(d.status('B'),'pending');assert.deepEqual(d.export().assignments,{});
  assert.deepEqual(d.export().exclude_windows,['B:0']);
});
test('applied human decisions reopen selected, including partial corrections',()=>{
  const data=make();data.applied_decisions={assignments:{B:'new:Child'},range_overrides:[{turn_id:'b',start:8,end:9,profile:'VOICE_0001'}]};
  const d=new ReviewDraft(data);assert.equal(d.groupChoice('B'),'new:Child');assert.deepEqual(d.clipChoices('B',7,10),['new:Child','VOICE_0001']);
});
test('export/import preserves all editable data',()=>{
  const d=new ReviewDraft(make());d.assignments.B='new:Child';d.updates.VOICE_0001={label:'Alex edited',role:'adult',archived:false};
  d.setRange('b',8,9,'VOICE_0001');d.excluded.add('A:0');const saved=d.export();
  const reopened=new ReviewDraft(make());reopened.load(JSON.parse(JSON.stringify(saved)));assert.deepEqual(reopened.export(),saved);
});
test('invalid and stale imports leave the current draft unchanged',()=>{
  const d=new ReviewDraft(make());d.assignments.B='new:Child';const before=d.export();
  for(const changes of [{review_id:'other'}, {packet:'other.json'}, {assignments:{B:'VOICE_9999'}},
                       {range_overrides:[{turn_id:'b',start:11,end:8,profile:'new:Child'}]}, {exclude_windows:['B:999']}]){
    assert.throws(()=>d.load({...before,...changes}));assert.deepEqual(d.export(),before);
  }
});
test('clip corrections split existing ranges without discarding the surrounding decisions',()=>{
  const d=new ReviewDraft(make());d.assignments.B='new:Child';d.setRange('b',7,12,'VOICE_0001');d.setClip('B',8,9,'new:Adult A');
  assert.deepEqual(d.ranges.map(r=>[r.start,r.end,r.profile]).sort((a,b)=>a[0]-b[0]),
    [[7,8,'VOICE_0001'],[8,9,'new:Adult A'],[9,12,'VOICE_0001']]);
});
test('deleting a draft person removes its choices and survives saved-draft import',()=>{
  const d=new ReviewDraft(make());d.assignments.B='new:Child';d.removeDraft('new:Child');
  assert.equal(d.assignments.B,'unknown');const saved=d.export();d.load(saved);assert.equal(d.newProfiles['new:Child'],undefined);
});
test('reimporting an already-created draft resolves to the original voice ID',()=>{
  const data=make();data.new_profile_ids={'new:Adult A':'VOICE_0001'};const d=new ReviewDraft(data);
  d.load({review_id:data.review_id,packet:data.packet,assignments:{B:'new:Adult A'},new_profiles:{'new:Adult A':{label:'Adult A',role:'adult'}}});
  assert.equal(d.assignments.B,'VOICE_0001');
});
test('complete snapshots can clear an earlier override',()=>{
  const data=make();data.applied_decisions={assignments:{B:'new:Child'},turn_overrides:{b:'VOICE_0001'}};
  const d=new ReviewDraft(data);delete d.overrides.b;const restored=new ReviewDraft(data);restored.load(d.export());
  assert.equal(restored.choiceAt(data.turns[1],8),'new:Child');
});
test('all inline review scripts parse without external dependencies',()=>{
  const html=fs.readFileSync('transcribe_media_app/review_page.html','utf8');
  for(const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
  assert.equal(/<script[^>]+src=/.test(html),false);
});

const makeBatch=()=>{const first=make(),second={...make(),review_id:'review-2',packet:'second.json',source:'second.wav'};
  return {kind:'speaker_review_batch',batch_id:'batch-identity',profiles:first.profiles,recordings:[first,second]};};
test('batch people and profile edits are shared but local assignments remain separate',()=>{
  const w=new ReviewWorkspace(makeBatch()),key=w.newKey();
  w.drafts[0].newProfiles[key]={label:'Shared person',role:'adult'};
  w.drafts[0].assignments.B=key;w.drafts[0].updates.VOICE_0001={label:'Alex edited',role:'adult'};
  w.synchronize(w.drafts[0]);
  assert.equal(w.drafts[1].label(key),'Shared person');assert.equal(w.drafts[1].label('VOICE_0001'),'Alex edited');
  assert.equal(w.drafts[1].groupChoice('B'),'');w.drafts[1].assignments.B=key;
  const saved=w.export();assert.equal(saved.reviews.length,2);assert.equal(saved.new_profiles[key].label,'Shared person');
  const restored=new ReviewWorkspace(makeBatch());restored.load(saved);assert.deepEqual(restored.export(),saved);
});
test('a malformed batch import changes no recording draft',()=>{
  const w=new ReviewWorkspace(makeBatch());w.drafts[0].assignments.B='ignore';const before=w.export();
  const bad=structuredClone(before);bad.reviews[1].review_id='other';
  assert.throws(()=>w.load(bad));assert.deepEqual(w.export(),before);
});
test('removing a shared draft person clears their decisions in every recording',()=>{
  const w=new ReviewWorkspace(makeBatch()),key=w.newKey();
  w.newProfiles[key]={label:'A person',role:'adult'};for(const d of w.drafts)d.assignments.B=key;
  w.removePerson(key);assert.ok(w.drafts.every(d=>d.groupChoice('B')==='unknown'));
  const saved=w.export();w.load(saved);assert.equal(w.newProfiles[key],undefined);
});
test('old batch imports resolve previously created people in a regenerated page',()=>{
  const input=makeBatch(),w=new ReviewWorkspace(input),key=w.newKey();
  w.newProfiles[key]={label:'Alex',role:'adult'};w.drafts[0].assignments.B=key;
  const saved=w.export();input.new_profile_ids={[key]:'VOICE_0001'};
  const updated=new ReviewWorkspace(input);updated.load(saved);assert.equal(updated.drafts[0].groupChoice('B'),'VOICE_0001');
});
test('importing one recording leaves other recording decisions intact',()=>{
  const w=new ReviewWorkspace(makeBatch());w.drafts[1].assignments.B='ignore';
  w.load({...make(),format_version:2,assignments:{B:'VOICE_0001'}});
  assert.equal(w.drafts[0].groupChoice('B'),'VOICE_0001');assert.equal(w.drafts[1].groupChoice('B'),'ignore');
});
test('a previously created single-recording draft person is reused throughout the batch',()=>{
  const data=makeBatch();data.recordings[0].new_profile_ids={'new:Adult A':'VOICE_0001'};
  const w=new ReviewWorkspace(data);
  w.load({format_version:2,packet:'packet.json',review_id:'review-1',assignments:{B:'new:Adult A'},
    new_profiles:{'new:Adult A':{label:'Adult A',role:'adult'}}});
  assert.equal(w.drafts[0].groupChoice('B'),'VOICE_0001');
  assert.equal(w.drafts[1].data.new_profile_ids['new:Adult A'],'VOICE_0001');
});
test('explicit unknown stays distinct from an unresolved review',()=>{
  const d=new ReviewDraft(make());d.assignments.B='ignore';assert.equal(d.status('B'),'reviewed');
  assert.match(d.label('ignore'),/UNKNOWN/);d.load(d.export());assert.equal(d.groupChoice('B'),'ignore');
  d.assignments.B='unknown';assert.equal(d.status('B'),'pending');
});
test('batch navigation includes uncertain timing until the affected turn is reviewed',()=>{
  const data=make();data.turns[0].speaker_attribution={status:'uncertain'};
  const d=new ReviewDraft(data);assert.equal(d.status('A'),'matched');assert.equal(d.needsAttention('A'),true);
  d.setRange('a',0,1,'ignore');assert.equal(d.needsAttention('A'),true);
  d.setRange('a',1,6,'VOICE_0001');assert.equal(d.needsAttention('A'),false);
});
test('apply command uses only the project decisions folder',()=>{
  assert.equal(ReviewIO.command('batch-123.decisions.json'),'./transcribe-media --apply-speaker-review "speaker-decisions/batch-123.decisions.json"');
  assert.throws(()=>ReviewIO.command('../secret.decisions.json'));
});
test('folder saving validates the project and writes the exact relative destination',async()=>{
  const calls=[];let content;
  const writer={write:async value=>content=value,close:async()=>calls.push('close'),abort:async()=>calls.push('abort')};
  const folder={getFileHandle:async(name,options)=>{calls.push([name,options]);return {createWritable:async()=>writer};}};
  const project={getFileHandle:async name=>calls.push(name),getDirectoryHandle:async(name,options)=>{calls.push([name,options]);return folder;}};
  const path=await ReviewIO.writeToProject(project,'batch-123.decisions.json',{example:true});
  assert.equal(path,'speaker-decisions/batch-123.decisions.json');assert.deepEqual(JSON.parse(content),{example:true});
  assert.deepEqual(calls,['transcribe-media',['transcribe_media_app',undefined],['speaker-decisions',{create:true}],['batch-123.decisions.json',{create:true}],'close']);
  await assert.rejects(()=>ReviewIO.writeToProject({getFileHandle:async()=>{throw Error('Wrong folder');}},'batch-123.decisions.json',{}));
});
