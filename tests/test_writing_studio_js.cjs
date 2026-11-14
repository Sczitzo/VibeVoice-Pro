const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync('demo/writing_studio.js', 'utf8');
function setup(value='',blocked=false) {
 const title=new EventTarget(); title.value='';
 const saved=new Map([['vibevoice-draft-title',value]]);
 const context={document:{getElementById:()=>title,querySelector:()=>null},sessionStorage:{getItem:k=>{if(blocked)throw Error('blocked');return saved.get(k);},setItem:(k,v)=>{if(blocked)throw Error('blocked');saved.set(k,v);}}};
 vm.runInNewContext(source+'\ninitWritingStudio();',context); return {title,saved,context};
}
test('Writing Studio restores session title without duplicate header',()=>{const h=setup('My session');assert.equal(h.title.value,'My session');h.title.value='Renamed';h.title.dispatchEvent(new Event('input'));assert.equal(h.saved.get('vibevoice-draft-title'),'Renamed');});
test('blocked session storage does not stop Composer',()=>{assert.doesNotThrow(()=>setup('',true));});
test('speaker highlighting escapes markup and preserves speaker label',()=>{const h=setup();const result=vm.runInNewContext('writingStudioMarkup(\'Speaker 2: <script>alert("x")</script> & text\')',h.context);assert.match(result,/Speaker 2:/);assert.match(result,/&lt;script&gt;/);assert.doesNotMatch(result,/<script>/);});
test('normal prose cannot create markup',()=>{const h=setup();const result=vm.runInNewContext('writingStudioMarkup(\'<img src=x onerror=alert(1)>\')',h.context);assert.equal(result,'&lt;img src=x onerror=alert(1)&gt;\n');});
test('empty editor markup remains empty content plus final line',()=>{const h=setup();assert.equal(vm.runInNewContext('writingStudioMarkup("")',h.context),'\n');});

test('cast disclosure preserves expansion, count, keyboard and focused resize',()=>{
 class El extends EventTarget { constructor(){super();this.attrs={};this.textContent='';this.classList={toggle:(k,v)=>this[k]=v};} setAttribute(k,v){this.attrs[k]=v;} focus(){doc.activeElement=this;} }
 const cast=new El(),count=new El(),button=new El(); let selected='4', mediaChange;
 cast.querySelector=()=>({value:selected});cast.prepend=()=>{};cast.contains=e=>e===count||e===button;
 const media={matches:true,addEventListener:(_,fn)=>{mediaChange=fn;}};
 const doc={activeElement:null,getElementById:id=>id==='composer-speakers'?cast:id==='speaker-count'?count:null,createElement:()=>button};
 const ctx={document:doc,window:{matchMedia:()=>media},MutationObserver:class{observe(){}}};
 vm.runInNewContext(source+'\ninitCastDisclosure();',ctx);
 assert.equal(button.attrs['aria-expanded'],'false');assert.match(button.textContent,/4/);
 button.dispatchEvent(new Event('click'));assert.equal(button.attrs['aria-expanded'],'true');
 selected='1';cast.dispatchEvent(new Event('change'));assert.match(button.textContent,/1/);
 const esc=new Event('keydown');esc.key='Escape';cast.dispatchEvent(esc);assert.equal(button.attrs['aria-expanded'],'false');assert.equal(doc.activeElement,button);
 media.matches=false;mediaChange();assert.equal(button.attrs['aria-expanded'],'true');
 doc.activeElement=count;media.matches=true;mediaChange();assert.equal(button.attrs['aria-expanded'],'true');assert.equal(doc.activeElement,count);
});

test('editor grows and shrinks with text, capped to remaining viewport',()=>{
 const h=setup(); const size=(n,a)=>vm.runInNewContext(`writingStudioEditorHeight(${n},${a})`,h.context);
 assert.equal(size(30,400),72); assert.equal(size(180,400),180);
 assert.equal(size(900,400),400); assert.equal(size(30,400),72);
 assert.equal(size(200,120),120); assert.equal(size(900,20),72);
 assert.equal(size(123.2,180.9),124);
});

test('autosize measures offscreen without touching text, caret or scroll and coalesces events',()=>{
 let frame, natural=180, bottom=500, paints=0;
 const props={}; const card={style:{getPropertyValue:k=>props[k],setProperty:(k,v)=>props[k]=v},getBoundingClientRect:()=>({bottom:300})};
 const measure={style:{},removeAttribute(){},setAttribute(){},get scrollHeight(){return natural;}};
 const editor=new EventTarget(); Object.assign(editor,{value:'Speaker 1: unchanged',placeholder:'',selectionStart:5,selectionEnd:9,scrollTop:41,cloneNode:()=>measure,getBoundingClientRect:()=>({top:100,bottom:250,width:400})});
 const grid={getBoundingClientRect:()=>({bottom})};
 const ctx={document:{getElementById:id=>id==='composer-grid'?grid:id==='composer-script'?card:null,body:{append(){}}},window:new EventTarget(),requestAnimationFrame:fn=>frame=fn,getComputedStyle:()=>({borderTopWidth:'0',borderBottomWidth:'0'})};
 vm.runInNewContext(source,ctx);ctx.initWritingStudioAutosize(editor,()=>paints++);frame();
 assert.equal(props['--editor-height'],'180px'); assert.equal(measure.value,editor.value);
 natural=900;editor.dispatchEvent(new Event('input'));editor.dispatchEvent(new Event('change'));frame();assert.equal(props['--editor-height'],'348px');
 natural=20;editor.dispatchEvent(new Event('input'));frame();assert.equal(props['--editor-height'],'72px');
 assert.equal(editor.value,'Speaker 1: unchanged'); assert.equal(editor.selectionStart,5);assert.equal(editor.selectionEnd,9);assert.equal(editor.scrollTop,41);assert.equal(paints,3);
});
