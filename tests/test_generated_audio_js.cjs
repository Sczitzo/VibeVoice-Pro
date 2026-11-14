const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync('demo/generated_audio.js', 'utf8');
function harness() {
    class Element extends EventTarget {
        constructor() { super(); this.attrs = {}; this.textContent = ''; this.disabled = false; }
        setAttribute(k,v) { this.attrs[k]=v; }
        getAttribute(k) { return k === 'src' ? this.src : this.attrs[k]; }
        hasAttribute(k) { return Boolean(this.getAttribute(k)); }
        removeAttribute(k) { delete this.attrs[k]; if(k==='src') this.src=''; }
    }
    class Audio extends Element {
        constructor() { super(); this.currentTime=0; this.duration=100; this.paused=true; this.readyState=0; this.ended=false; }
        play() { this.paused=false; this.dispatchEvent(new Event('play')); return Promise.resolve(); }
        pause() { this.paused=true; this.dispatchEvent(new Event('pause')); }
        load() {}
    }
    const audio=new Audio(), live=new Audio();
    const ids = Object.fromEntries(['playback-toggle','playback-waveform','playback-seek','playback-title','playback-elapsed','playback-position-time','playback-duration','generation-progress','download-wav'].map(k=>[k,new Element()]));
    ids['playback-waveform'].getContext=()=>null;
    const replay=new Element(), clear=new Element(), stop={visible:false,getClientRects(){return this.visible?[{}]:[];}};
    const completed={querySelector:s=>s==='audio'?audio:s==='[data-replay-audio]'?replay:s==='[data-clear-audio]'?clear:ids[s.slice(1)]};
    const liveRegion=new Element(); liveRegion.querySelector=()=>live;
    const composer={querySelector:s=>s==='#stop-generation'?stop:null};
    let sync, captureClick;
    const window=new EventTarget();window.location={href:'http://localhost:7881/',origin:'http://localhost:7881'};
    const context={window,URL,AbortController,Event,Float32Array,Uint8Array,DataView,fetch:async()=>({ok:false}),
        MutationObserver:class{constructor(fn){sync=fn;} observe(){}},
        document:{addEventListener:(event,fn)=>{captureClick=fn;},getElementById:id=>({'guided-composer':composer,'live-generated-audio':liveRegion,'generated-audio-player':completed,'composer-actions':{}}[id])}};
    vm.runInNewContext(source+'\ninitGeneratedAudio();',context);
    const begin=()=>{window.vibevoicePlayback.start();stop.visible=true;live.src='/new-stream';live.readyState=2;sync();};
    const finish=(url='/final.wav')=>{stop.visible=false;sync();window.vibevoicePlayback.complete({url});audio.readyState=1;audio.dispatchEvent(new Event('loadedmetadata'));};
    return {audio,live,clear,stop,sync,begin,finish,submit:(target="generate")=>{let blocked=false;captureClick({target:{closest:selector=>target==="section" ? selector.includes("data-section-command") : true},preventDefault(){blocked=true;},stopImmediatePropagation(){}});return !blocked;},play:ids['playback-toggle'],api:window.vibevoicePlayback, meter:ids['generation-progress'], duration:ids['playback-duration']};
}
test('estimated progress and ETA update while playback is paused and clear on Stop', () => {
    const h = harness();
    h.begin();
    h.play.dispatchEvent(new Event('click'));
    h.api.finish({running:true, progress:{seconds:8, percent:40, eta:12}}, null);
    assert.equal(h.live.paused, true);
    assert.equal(h.duration.textContent, '~40% · ~0:12 left');
    assert.equal(h.meter.max, 100);
    assert.equal(h.meter.value, 40);
    assert.equal(h.meter.hidden, false);
    h.api.stop();
    h.api.finish({running:true, progress:{seconds:10}}, null);
    assert.equal(h.meter.hidden, true);
});
test('paused live audio transfers elapsed position to final WAV without stopping generation',async()=>{
    const h=harness();h.begin();await Promise.resolve();h.live.currentTime=18;
    h.play.dispatchEvent(new Event('click'));
    assert.equal(h.live.paused,true);assert.equal(h.stop.visible,true);
    h.finish();assert.equal(h.audio.currentTime,18);assert.equal(h.audio.paused,true);
    h.play.dispatchEvent(new Event('click'));assert.equal(h.audio.currentTime,18);assert.equal(h.audio.paused,false);assert.equal(h.play.textContent,'Pause');
});
test('playing live audio continues final WAV at the same position',async()=>{
    const h=harness();h.begin();await Promise.resolve();h.live.currentTime=22;
    h.finish();assert.equal(h.audio.currentTime,22);assert.equal(h.audio.paused,false);assert.equal(h.play.textContent,'Pause');
});
test('explicit Stop retains old completed take and ignores late completion; export resets safely',()=>{
    const h=harness();h.api.complete({url:'/old.wav'});h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));
    h.begin();h.api.stop();h.api.complete({url:'/late.wav'});assert.equal(h.audio.src,'http://localhost:7881/old.wav');
    h.api.acceptExport();h.api.complete({url:'/export.wav'});assert.equal(h.audio.src,'http://localhost:7881/export.wav');assert.equal(h.audio.paused,true);
});
test('late metadata cannot resume a completion after a new generation starts',()=>{
    const h=harness();h.begin();h.live.currentTime=5;h.api.complete({url:'/final.wav'});h.api.start();h.audio.dispatchEvent(new Event('loadedmetadata'));assert.equal(h.audio.paused,true);
});

test('completion before Stop control hides still preserves position and paused state',()=>{
    const h=harness();h.begin();h.live.currentTime=31;h.play.dispatchEvent(new Event('click'));
    h.api.complete({url:'/final.wav'});h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));
    h.stop.visible=false;h.sync();assert.equal(h.audio.currentTime,31);assert.equal(h.audio.paused,true);
});
test('handoff at or past final duration stays finished instead of restarting',()=>{
    const h=harness();h.begin();h.live.currentTime=120;h.live.ended=true;h.live.paused=true;
    h.finish();assert.equal(h.audio.currentTime,100);assert.equal(h.audio.paused,true);assert.equal(h.play.textContent,'Play');
});
test('completion with no live audio remains paused at beginning',()=>{
    const h=harness();h.api.start();h.stop.visible=true;h.sync();h.finish();
    assert.equal(h.audio.currentTime,0);assert.equal(h.audio.paused,true);
});

test('clearing a newly completed take cancels delayed resume',()=>{
    const h=harness();h.begin();h.live.currentTime=4;h.api.complete({url:'/final.wav'});
    h.clear.dispatchEvent(new Event('click'));h.audio.dispatchEvent(new Event('loadedmetadata'));
    assert.equal(h.audio.src,'');assert.equal(h.audio.paused,true);
});

test('final Play waits for metadata and its transferred seek position',()=>{
    const h=harness();h.begin();h.live.currentTime=9;h.play.dispatchEvent(new Event('click'));
    h.api.complete({url:'/final.wav'});assert.equal(h.play.disabled,true);
    h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));
    assert.equal(h.play.disabled,false);assert.equal(h.audio.currentTime,9);assert.equal(h.audio.paused,true);
});

test('completed-outcome handoff never exposes previous audio during callback gap',()=>{
    const h=harness();h.api.complete({url:'/old.wav'});h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));
    h.begin();h.live.currentTime=7;h.play.dispatchEvent(new Event('click'));
    h.stop.visible=false;h.sync();assert.equal(h.play.disabled,true);
    h.play.dispatchEvent(new Event('click'));assert.equal(h.audio.paused,true);
    h.api.finish({completed:true,running:false},{url:'/new.wav'});
    h.audio.dispatchEvent(new Event('loadedmetadata'));assert.equal(h.audio.currentTime,7);assert.equal(h.audio.paused,true);
});
test('failed outcome releases prior completed audio without transferring failed live position',()=>{
    const h=harness();h.api.complete({url:'/old.wav'});h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));
    h.audio.currentTime=3;h.begin();h.live.currentTime=8;h.stop.visible=false;h.sync();
    h.api.finish({completed:false,running:false},null);
    assert.equal(h.audio.currentTime,3);assert.equal(h.audio.src,'http://localhost:7881/old.wav');assert.equal(h.play.disabled,false);
});

test('new Generate cannot submit during final-file gap and failure releases the gate',()=>{
    const h=harness();assert.equal(h.submit(),true);h.begin();h.stop.visible=false;h.sync();
    assert.equal(h.api.isPending(),true);assert.equal(h.submit(),false);
    h.api.finish({running:false,completed:false},null);
    assert.equal(h.api.isPending(),false);assert.equal(h.submit(),true);
});

test('Stop retains submission gate until worker outcome while previous take remains playable',()=>{
    const h=harness();h.api.complete({url:'/old.wav'});h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));
    h.begin();h.api.stop();assert.equal(h.submit(),false);assert.equal(h.play.disabled,false);
    h.api.finish({running:false,completed:false},null);assert.equal(h.submit(),true);
});

test('retained section commands cannot bypass a pending generation gate',()=>{
    const h=harness();h.begin();assert.equal(h.submit('section'),false);
    h.api.finish({running:false,completed:false},null);assert.equal(h.submit('section'),true);
});
test('repeated same-file export releases pending without resetting position',()=>{
    const h=harness();h.api.complete({url:'/saved.wav'});h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));h.audio.currentTime=11;
    h.api.acceptExport();h.api.finish({running:false,completed:true},{url:'/saved.wav'});
    assert.equal(h.api.isPending(),false);assert.equal(h.audio.currentTime,11);assert.equal(h.submit(),true);
});
test('invalid terminal file releases pending and preserves prior valid take',()=>{
    const h=harness();h.api.complete({url:'/old.wav'});h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));
    h.begin();h.api.finish({running:false,completed:true},{url:'https://other.example/wrong.wav'});
    assert.equal(h.api.isPending(),false);assert.equal(h.audio.src,'http://localhost:7881/old.wav');assert.equal(h.play.disabled,false);
});

test('serialized terminal outcome completes handoff and malformed outcome safely releases',()=>{
 const h=harness();h.begin();h.live.currentTime=6;h.api.finish(JSON.stringify({running:false,completed:true}),{url:'/new.wav'});
 h.audio.readyState=1;h.audio.dispatchEvent(new Event('loadedmetadata'));assert.equal(h.audio.currentTime,6);assert.equal(h.api.isPending(),false);
 h.api.start();h.api.finish('invalid',null);assert.equal(h.api.isPending(),false);
});
