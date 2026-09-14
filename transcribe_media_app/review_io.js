/* Browser file access is optional. A download never reveals its destination. */
(function(root){
  'use strict';
  const directory='speaker-decisions';
  function validName(name){
    if(!/^[a-zA-Z0-9_-]+\.decisions\.json$/.test(name))throw Error('Invalid decision filename.');
    return name;
  }
  function relativePath(name){return directory+'/'+validName(name);}
  function command(name){return './transcribe-media --apply-speaker-review "'+relativePath(name)+'"';}
  function folderSupport(environment){
    return environment.isSecureContext === true && typeof environment.showDirectoryPicker === 'function';
  }
  function folderFailure(error){
    if(error.name==='AbortError')return {blocked:false,text:'Folder saving cancelled. Your draft is still available; use Download JSON when ready.'};
    if(['SecurityError','NotAllowedError'].includes(error.name))return {blocked:true,
      text:'This browser blocked folder access. Use Download JSON, move the file into speaker-decisions/ in your project, then run the copied apply command.'};
    return {blocked:false,text:'Could not save to that folder. Select the project containing transcribe-media, or use Download JSON and move it into speaker-decisions/.'};
  }
  async function writeToProject(handle,name,payload){
    // Reject an accidental selection such as Downloads before creating anything.
    await handle.getFileHandle('transcribe-media');
    await handle.getDirectoryHandle('transcribe_media_app');
    const folder=await handle.getDirectoryHandle(directory,{create:true});
    const file=await folder.getFileHandle(validName(name),{create:true});
    const writer=await file.createWritable();
    try{await writer.write(JSON.stringify(payload,null,2)+'\n');await writer.close();}
    catch(error){try{await writer.abort();}catch{}throw error;}
    return relativePath(name);
  }
  root.ReviewIO={directory,relativePath,command,writeToProject,folderSupport,folderFailure};
  if(typeof module!=='undefined')module.exports=root.ReviewIO;
})(globalThis);
