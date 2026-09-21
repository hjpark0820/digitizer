/* Browser-owned folder handles only: the server never receives a writable path. */
(function(root) {
  'use strict';
  const FORMAT = 'chartocode-v46-image-sidecar-v1';
  const HISTORY_FORMAT = 'chartocode-v46-image-history-v2';
  const MAX_JSON = 128*1024*1024;
  const IMAGE = /\.(png|jpe?g)$/i;
  function stem(name) {
    if (!IMAGE.test(name) || /[/\\]/.test(name)) throw new Error('Select a PNG/JPG image.');
    return name.replace(IMAGE,'');
  }
  async function digest(file) {
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())))
      .map(v=>v.toString(16).padStart(2,'0')).join('');
  }
  async function optionalFile(directory,name) {
    try { return await (await directory.getFileHandle(name)).getFile(); }
    catch(error) { if(error.name==='NotFoundError')return null; throw error; }
  }
  function stable(value) {
    if(Array.isArray(value))return '['+value.map(stable).join(',')+']';
    if(value && typeof value==='object')return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+stable(value[k])).join(',')+'}';
    return JSON.stringify(value);
  }
  async function hashText(text) { return digest(new Blob([text])); }
  function validateHistory(value) {
    if(value?.format!==HISTORY_FORMAT || !value.source_image?.file_sha256 ||
       !Array.isArray(value.versions) || !value.versions.length || value.versions.length>1000 ||
       !value.objects || typeof value.objects!=='object')throw new Error('Invalid result history JSON.');
    const seen=new Set();
    for(const v of value.versions) {
      if(typeof v.id!=='string' || !/^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,100}$/.test(v.id) || seen.has(v.id) ||
         (v.parent_id!=null && !seen.has(v.parent_id)) || v.snapshot?.format!==FORMAT ||
         !Array.isArray(v.snapshot.edit_data?.curves))throw new Error('Invalid result version or parent.');
      for(const ref of Object.values(v.snapshot.artifact_refs||{}))if(typeof value.objects[ref]!=='string')throw new Error('Missing saved evidence.');
      if(v.snapshot.working_image_ref && typeof value.objects[v.snapshot.working_image_ref]!=='string')throw new Error('Missing prepared image.');
      seen.add(v.id);
    }
    if(!seen.has(value.selected_version_id))throw new Error('Selected version is missing.');
    return value;
  }
  async function asHistory(value) {
    if(value?.format===HISTORY_FORMAT) {
      validateHistory(value);
      for(const [key,raw] of Object.entries(value.objects))if(key!==await hashText(raw))throw new Error('Saved evidence checksum mismatch.');
      return structuredClone(value);
    }
    if(value?.format!==FORMAT || !value.source_image?.file_sha256)throw new Error('Not a linked correction JSON.');
    const snapshot=structuredClone(value), objects={}; delete snapshot.source_image;
    async function store(obj) { const raw=JSON.stringify(obj),key=await hashText(raw);objects[key]=raw;return key; }
    snapshot.artifact_refs={};
    for(const [name,obj] of Object.entries(snapshot.artifacts||{}))snapshot.artifact_refs[name]=await store(obj);
    delete snapshot.artifacts;
    if(snapshot.working_image_png_base64) {
      snapshot.working_image_ref=await store(snapshot.working_image_png_base64);delete snapshot.working_image_png_base64;
    }
    const id='legacy_'+(await hashText(stable(value))).slice(0,24);
    return validateHistory({format:HISTORY_FORMAT,source_image:structuredClone(value.source_image),objects,
      selected_version_id:id,versions:[{id,parent_id:null,kind:'imported',label:'Previously saved result',
        created_at:null,iterations:null,point_count:(snapshot.edit_data?.curves||[]).reduce((n,c)=>n+c.points.length,0),snapshot}]});
  }
  function versionPackage(history,id) {
    validateHistory(history);
    const v=history.versions.find(v=>v.id===(id||history.selected_version_id));
    if(!v)throw new Error('Selected version is missing.');
    const packet=structuredClone(v.snapshot);
    packet.source_image=structuredClone(history.source_image);packet.artifacts={};
    for(const [name,ref] of Object.entries(packet.artifact_refs||{}))packet.artifacts[name]=JSON.parse(history.objects[ref]);
    delete packet.artifact_refs;
    if(packet.working_image_ref) { packet.working_image_png_base64=JSON.parse(history.objects[packet.working_image_ref]);delete packet.working_image_ref; }
    return packet;
  }
  async function mergeHistories(previous,incoming) {
    const next=await asHistory(incoming);
    if(!previous)return next;
    const merged=await asHistory(previous);
    if(merged.source_image.file_sha256!==next.source_image.file_sha256)throw new Error('Different source image in result history.');
    for(const v of next.versions) {
      const old=merged.versions.find(row=>row.id===v.id);
      if(old) {
        if(stable(old.snapshot)!==stable(v.snapshot) || old.parent_id!==v.parent_id)throw new Error('Saved version conflict; previous JSON was not changed.');
      } else merged.versions.push(v);
    }
    for(const [key,raw] of Object.entries(next.objects)) {
      if(key in merged.objects && merged.objects[key]!==raw)throw new Error('Saved evidence conflict.');
      merged.objects[key]=raw;
    }
    merged.selected_version_id=next.selected_version_id;
    return validateHistory(merged);
  }
  async function readPreview(imageFile,jsonFile) {
    if(!jsonFile)throw new Error('The matching JSON is missing.');
    if(jsonFile.size>MAX_JSON)throw new Error('Correction JSON exceeds 128 MB.');
    const value=JSON.parse((await jsonFile.text()).replace(/^\uFEFF/,''));
    let packet=value, source=value?.source_image, version='Saved result', workingImage=null, sourceVerified=false;
    if(value?.format===HISTORY_FORMAT) {
      // Preview only the selected snapshot. Correction evidence is validated in
      // full on Open; browsing need not clone/decode every saved artifact.
      validateHistory(value);
      const index=value.versions.findIndex(v=>v.id===value.selected_version_id);
      const selected=value.versions[index];packet=selected.snapshot;
      const labels={detection:'Initial detection',step5:'Step 5 correction',manual:'Manual edits',imported:'Previously saved result'};
      version=(labels[selected.kind]||selected.label||'Saved result')+' · version '+(index+1)+' / '+value.versions.length;
      if(packet.working_image_ref) {
        const raw=value.objects[packet.working_image_ref];
        if(await hashText(raw)!==packet.working_image_ref)throw new Error('Saved image checksum mismatch.');
        workingImage=JSON.parse(raw);
      }
    } else if(value?.format && ![FORMAT,'chartocode-v46-correction-session-v1'].includes(value.format)) {
      throw new Error('Unsupported saved result format.');
    }
    if(packet?.format===FORMAT) {
      if(!source?.file_sha256 || await digest(imageFile)!==source.file_sha256)
        throw new Error('Original image file does not match the saved result.');
      sourceVerified=true;
      workingImage=workingImage || packet.working_image_png_base64 || null;
    }
    if(workingImage!==null && typeof workingImage!=='string')throw new Error('Invalid saved working image.');
    const editData=packet?.format ? packet.edit_data : packet;
    if(!editData || !Array.isArray(editData.curves))throw new Error('No editable curves in the saved JSON.');
    return {editData,workingImage,version,sourceVerified};
  }
  class ImageSidecar {
    constructor() { this.directory=null; this.source=null; this.sourceDigest=null; }
    async select(directory) {
      const files=new Set();
      for await (const [name,entry] of directory.entries()) {
        if(entry.kind==='file')files.add(name);
      }
      const images=[...files].filter(name=>IMAGE.test(name) && files.has(stem(name)+'.json'));
      images.sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
      this.directory=directory;
      return images;
    }
    async image(name) {
      if(!this.directory)throw new Error('Choose the image folder first.');
      stem(name);
      return (await this.directory.getFileHandle(name)).getFile();
    }
    async bind(file, valid=()=>true) {
      const hash=await digest(file);
      if(!valid())return false;
      this.source=file; this.sourceDigest=hash;
      return true;
    }
    async pair(name) {
      if(!this.directory)throw new Error('Choose the image folder first.');
      stem(name);
      const image=await (await this.directory.getFileHandle(name)).getFile();
      const json=await optionalFile(this.directory,stem(name)+'.json');
      return {image,json};
    }
    async matchingPair(file) {
      if(!this.directory)return null;
      const disk=await optionalFile(this.directory,file.name);
      if(!disk || await digest(disk)!==await digest(file))return null;
      return this.pair(file.name);
    }
    async save(packageValue) {
      if(!this.directory || !this.source)throw new Error('Choose the folder containing the original image.');
      const name=stem(this.source.name)+'.json';
      if(![FORMAT,HISTORY_FORMAT].includes(packageValue.format) || packageValue.source_image?.file_sha256!==this.sourceDigest)
        throw new Error('Exported JSON does not match the loaded original image.');
      const current=await optionalFile(this.directory,this.source.name);
      if(!current || await digest(current)!==this.sourceDigest)
        throw new Error('This folder does not contain the unchanged original image. No file was saved.');
      const existing=await optionalFile(this.directory,name);
      let previous=null;
      if(existing) {
        if(existing.size>MAX_JSON)throw new Error('Existing JSON exceeds 128 MB; it was not overwritten.');
        try { previous=JSON.parse(await existing.text()); } catch(_) {}
        if(![FORMAT,HISTORY_FORMAT].includes(previous?.format) || previous.source_image?.file_sha256!==this.sourceDigest)
          throw new Error(name+' already exists and is not this image’s linked correction JSON. It was not overwritten.');
      }
      const history=await mergeHistories(previous,packageValue);
      const text=JSON.stringify(history);
      if(new Blob([text]).size>MAX_JSON)throw new Error('Result history exceeds 128 MB. Previous JSON was not changed.');
      const handle=await this.directory.getFileHandle(name,{create:true});
      const writer=await handle.createWritable();
      try { await writer.write(text); await writer.close(); }
      catch(error) { try { await writer.abort(); } catch(_) {} throw error; }
      return name;
    }
  }
  root.ImageSidecarV46={ImageSidecar,stem,digest,asHistory,versionPackage,mergeHistories,readPreview,HISTORY_FORMAT,MAX_JSON};
  if(typeof module!=='undefined')module.exports=root.ImageSidecarV46;
})(globalThis);
