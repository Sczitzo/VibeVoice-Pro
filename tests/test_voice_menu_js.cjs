const test=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');
test('voice results and catalogs observe in-place Gradio attribute updates',()=>{
 const menu=fs.readFileSync('demo/voice_menu.js','utf8'); const composer=fs.readFileSync('demo/composer.js','utf8');
 assert.match(menu,/observe\(resultField,\s*\{[^}]*attributes:\s*true/);
 assert.match(menu,/observe\(catalog,\s*\{[^}]*attributes:\s*true/);
 assert.match(composer,/observe\(catalog,\s*\{[^}]*attributes:\s*true/);
});
const vm=require('node:vm');
test('preset grouping keeps custom options and reveals protected names for keyboard search',()=>{
 let state='true',handler,inputHandler;const attrs={};
 const input={getAttribute:()=>state,setAttribute:(k,v)=>attrs[k]=v,matches:()=>true,closest:()=>item};
 const rows=['Alice','Custom'].map(name=>({name,getAttribute:()=>name,classList:{toggle(){}},hidden:false}));
 let button;
 const list={querySelectorAll:()=>rows,classList:{add(){},remove(){}},after:b=>button=b,getBoundingClientRect:()=>({left:10,top:20,width:200})};
 const item={querySelector:s=>s==='ul[role="listbox"]'?list:s==='.preset-disclosure'?button:input};
 const assignments={querySelectorAll:()=>[item],addEventListener:(type,fn)=>{if(type==='keydown')handler=fn;if(type==='input')inputHandler=fn;}};
 const catalog={querySelector:()=>({dataset:{builtinVoiceCatalog:'["Alice"]'}})};
 const ctx={document:{createElement:()=>({attrs:{},style:{},addEventListener(){},getAttribute(k){return this.attrs[k]},setAttribute(k,v){this.attrs[k]=v},remove(){button=null}})},MutationObserver:class{constructor(fn){this.fn=fn;fn()}observe(){}},window:{addEventListener(){}}};
 vm.runInNewContext(fs.readFileSync('demo/voice_menu.js','utf8'),ctx);ctx.initPresetGroups(assignments,catalog);
 assert.equal(rows[0].hidden,true);assert.equal(rows[1].hidden,false);assert.equal(button.tabIndex,-1);
 handler({target:input,key:'ArrowRight',preventDefault(){},stopImmediatePropagation(){}});assert.equal(rows[0].hidden,false);
 handler({target:input,key:'ArrowLeft',preventDefault(){},stopImmediatePropagation(){}});assert.equal(rows[0].hidden,true);
 inputHandler({target:input});assert.equal(rows[0].hidden,false);
});
test('result freshness is tied to response identity, not its message',()=>{
 const ctx={};vm.runInNewContext(fs.readFileSync('demo/voice_menu.js','utf8'),ctx);
 assert.equal(ctx.voiceMenuResultIsNew('old','old'),false);
 assert.equal(ctx.voiceMenuResultIsNew('new','old'),true);
 assert.equal(ctx.voiceMenuResultIsNew(undefined,'old'),false);
});
test('cross-tab refresh ignores the retained prior catalog render',()=>{
 const source=fs.readFileSync('demo/composer.js','utf8');
 assert.match(source,/priorCatalogUpdate = catalog\.querySelector/);
 assert.match(source,/dataset\.update === priorCatalogUpdate\) return/);
});
