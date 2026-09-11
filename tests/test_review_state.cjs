const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {ReviewDraft}=require('../transcribe_media_app/review_state.js');
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
