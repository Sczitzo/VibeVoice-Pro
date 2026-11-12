const assert=require('node:assert/strict');
const {test}=require('node:test');
const fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('demo/composer.js','utf8');
function harness(){
 class El extends EventTarget{constructor(){super();this.dataset={};this.value='';this.style={setProperty(key,value){this[key]=value;}};}getBoundingClientRect(){return {width:100};}setAttribute(){}focus(){this.dispatchEvent(new Event('focus'));}append(){}querySelectorAll(){return [];}querySelector(){return new El();}}
 const editor=new El();editor.value='My draft';editor.selectionStart=editor.selectionEnd=8;
 editor.setSelectionRange=(s,e)=>{editor.selectionStart=s;editor.selectionEnd=e;};
 const undo=new El();undo.hidden=true;
 const count=new El();const radios=[1,2,3,4].map(n=>({value:String(n),checked:n===2,click(){radios.forEach(r=>r.checked=r===this);count.dispatchEvent(new Event('change'));}}));
 count.querySelector=()=>radios.find(r=>r.checked);count.querySelectorAll=()=>radios;
 const elements={'guided-composer':new El(),'speaker-count':count,'undo-script':undo};
 const document={querySelector:()=>editor,querySelectorAll:()=>[],getElementById:id=>elements[id]||(elements[id]=new El())};
 const window={};vm.runInNewContext(source+'\ninitComposer();',{document,window,Event,Intl});
 const type=text=>{editor.value=text;editor.selectionStart=editor.selectionEnd=text.length;editor.dispatchEvent(new Event('input'));};
 const key=(key,shift=false)=>{const e=new Event('keydown',{cancelable:true});Object.assign(e,{key,ctrlKey:true,shiftKey:shift});editor.dispatchEvent(e);return e.defaultPrevented;};
 return {elements,editor,undo,type,key,example:window.vibevoiceComposer.applyExample,count,radios};
}
test('typing shows Undo and restores prior text with keyboard redo',()=>{
 const h=harness();assert.equal(h.undo.hidden,true);h.type('Changed draft');assert.equal(h.undo.hidden,false);
 assert.equal(h.key('z'),true);assert.equal(h.editor.value,'My draft');assert.equal(h.undo.hidden,true);
 h.key('z',true);assert.equal(h.editor.value,'Changed draft');
});
test('cycling examples restores the entire overwritten draft and speaker count in order',()=>{
 const h=harness();h.type('Important edited draft');h.example({script:'Example one',speakers:1});h.example({script:'Example two',speakers:4});
 h.undo.dispatchEvent(new Event('click'));assert.equal(h.editor.value,'Example one');assert.equal(h.count.querySelector().value,'1');
 h.undo.dispatchEvent(new Event('click'));assert.equal(h.editor.value,'Important edited draft');assert.equal(h.count.querySelector().value,'2');
 h.undo.dispatchEvent(new Event('click'));assert.equal(h.editor.value,'My draft');
});
test('new edit after Undo branches history and discards stale redo',()=>{
 const h=harness();h.example({script:'Example',speakers:1});h.key('z');h.type('New direction');h.key('z',true);assert.equal(h.editor.value,'New direction');
});
test('example captures edits made while awaiting the response and browser Undo menu works',()=>{
 const h=harness();h.type('Typed while waiting');h.example({script:'Loaded example',speakers:3});
 const e=new Event('beforeinput',{cancelable:true});e.inputType='historyUndo';h.editor.dispatchEvent(e);
 assert.equal(e.defaultPrevented,true);assert.equal(h.editor.value,'Typed while waiting');
});

test('serialized JSON transport applies example and remains undoable',()=>{
 const h=harness();h.example(JSON.stringify({script:'Transport example',speakers:3}));
 assert.equal(h.editor.value,'Transport example');h.key('z');assert.equal(h.editor.value,'My draft');
});

test('Undo restores pre-edit selection after the user moves the caret',()=>{
 const h=harness();h.editor.setSelectionRange(2,5);
 const before=new Event('beforeinput');before.inputType='insertText';h.editor.dispatchEvent(before);
 h.type('My changed draft');h.key('z');
 assert.equal(h.editor.value,'My draft');assert.equal(h.editor.selectionStart,2);assert.equal(h.editor.selectionEnd,5);
});

test('panel border cutouts use the measured heading width',()=>{
 const h=harness();
 assert.equal(h.elements['composer-script'].style['--heading-gap'],'100px');
 assert.equal(h.elements['composer-speakers'].style['--heading-gap'],'100px');
});
