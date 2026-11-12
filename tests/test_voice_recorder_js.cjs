const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../demo/voice_recorder.js"), "utf8");
const audio = {
    numberOfChannels: 2, length: 3, sampleRate: 44100,
    getChannelData: channel => channel ? [0.5, -0.5, 0] : [-1, 1, 2]
};

function harness() {
    const sourceModes = {};
    const inputFile = Object.assign(new EventTarget(), {files: [], focus() {}});
    const importStart = Object.assign(new EventTarget(), {valueAsNumber: 0});
    const importEnd = Object.assign(new EventTarget(), {valueAsNumber: 1});
    const importRange = {querySelector: selector => selector === "canvas" ? {
        width: 4, height: 72, getContext: () => ({clearRect() {}, fillRect() {}})
    } : {}};
    const record = new EventTarget();
    const stop = new EventTarget();
    const pause = new EventTarget();
    const open = new EventTarget();
    const close = new EventTarget();
    const done = Object.assign(new EventTarget(), {focus() { this.focused = true; }});
    const savedDownload = {removeAttribute(key) { delete this[key]; }};
    const cancel = new EventTarget();
    const elapsed = {};
    const readingTitle = {focus() { this.focused = true; }};
    const save = new EventTarget();
    save.classList = {add() {}};
    const microphone = {value: "", focus() {}, replaceChildren(...options) { this.options = options; }};
    const refresh = new EventTarget();
    const microphoneInfo = {};
    const level = {value: 0};
    const levelMessage = {};
    const preview = {
        pause() {}, focus() {}, load() {},
        removeAttribute(key) { delete this[key]; }
    };
    const status = {toggleAttribute() {}};
    const name = Object.assign(new EventTarget(), {focus() {}});
    const saveResult = {setAttribute() {}};
    const savePanel = {querySelector: selector => selector === "#voice-save-status" ? saveResult : name};
    open.focus = stop.focus = () => {};
    const controls = Object.fromEntries(
        ["[data-retake]", "[data-return]", "[data-name]", "[data-back]"].map(key => [key, new EventTarget()])
    );
    const root = Object.assign(new EventTarget(), {
        dataset: {}, open: false, toggleAttribute() {}, removeAttribute() {},
        showModal() { this.open = true; },
        close() { this.open = false; },
        querySelectorAll: () => [],
        querySelector: selector => ({"[data-record]": record, "[data-stop]": stop,
            "[data-pause]": pause, "[data-close]": close, "[data-elapsed]": elapsed,
            "[data-done]": done, "[data-saved-download]": savedDownload,
            "#voice-reading-title": readingTitle,
            "[data-cancel]": cancel, "[data-capture-state]": {}, "[data-active-device]": {},
            "[data-save-controls]": {append() {}}, ...controls,
            "audio": preview, "[data-recording-status]": status,
            "[data-microphone]": microphone, "[data-refresh]": refresh,
            "[data-microphone-info]": microphoneInfo, "[data-level]": level,
            "[data-level-message]": levelMessage, ".voice-modes": sourceModes, "#voice-import-file": inputFile,
            "[data-import-range]": importRange, "[data-import-start]": importStart,
            "[data-import-end]": importEnd})[selector]
    });
    const metrics = {uploads: 0, decodes: 0, closed: 0, stopped: 0, revoked: 0,
        meterClosed: 0, signal: 0, now: 0, constraints: null, disconnected: 0};
    const frames = new Map();
    let frameId = 0;
    const track = new EventTarget();
    track.label = "Example USB Microphone";
    track.getSettings = () => ({sampleRate: 44100, deviceId: "wireless"});
    const stream = {
        getTracks: () => [{stop: () => metrics.stopped++}],
        getAudioTracks: () => [track]
    };
    class Recorder extends EventTarget {
        constructor(input) { super(); this.stream = input; this.mimeType = "audio/webm"; }
        start() { this.state = "recording"; }
        pause() { this.state = "paused"; }
        resume() { this.state = "recording"; }
        stop() {
            this.state = "inactive";
            const data = new Event("dataavailable");
            data.data = new Blob(["fixture recording"]);
            this.dispatchEvent(data);
            this.dispatchEvent(new Event("stop"));
        }
    }
    const window = new EventTarget();
    window.gradio_config = {root: "http://localhost:7865"};
    const context = vm.createContext({
        Blob, FormData, AbortController, ArrayBuffer, DataView, TextDecoder, Event, MediaRecorder: Recorder,
        window,
        document: {
            getElementById: id => ({"voice-recorder": root, "save-voice-btn": save, "open-voice-recorder": open,
                "voice-save-panel": savePanel})[id],
            querySelector: () => ({setAttribute() {}})
        },
        navigator: {mediaDevices: {getUserMedia: async () => stream}},
        Option: class { constructor(text, value) { this.text = text; this.value = value; } },
        performance: {now: () => metrics.now},
        requestAnimationFrame: callback => { frames.set(++frameId, callback); return frameId; },
        cancelAnimationFrame: id => frames.delete(id),
        URL: {createObjectURL: () => "blob:fixture", revokeObjectURL: () => metrics.revoked++},
        AudioContext: class {
            constructor(options) { this.meter = !options; this.state = "running"; }
            async resume() { this.state = "running"; }
            createMediaStreamSource() {
                return {connect() {}, disconnect: () => metrics.disconnected++};
            }
            createAnalyser() {
                return {getFloatTimeDomainData: samples => samples.fill(metrics.signal)};
            }
            async decodeAudioData() { metrics.decodes++; return audio; }
            async close() { this.state = "closed"; this.meter ? metrics.meterClosed++ : metrics.closed++; }
        },
        fetch: async () => {
            metrics.uploads++;
            return {ok: true, json: async () => ["/uploaded/voice.wav"]};
        },
        setTimeout: (callback, delay) => setTimeout(callback, delay === 20000 ? 15 : delay),
        clearTimeout, setInterval: () => 1, clearInterval() {}
    });
    const mediaDevices = context.navigator.mediaDevices;
    mediaDevices.enumerateDevices = async () => [
        {kind: "audioinput", deviceId: "default", label: "Default"},
        {kind: "audioinput", deviceId: "virtual", label: "Virtual microphone"},
        {kind: "audioinput", deviceId: "wireless", label: "Example USB Microphone"},
        {kind: "audiooutput", deviceId: "speaker", label: "Speakers"}
    ];
    mediaDevices.addEventListener = () => {};
    mediaDevices.getUserMedia = async constraints => { metrics.constraints = constraints; return stream; };
    vm.runInContext(source + "\ninitVoiceRecorder();", context);
    const capture = async () => {
        record.dispatchEvent(new Event("click"));
        await new Promise(resolve => setImmediate(resolve));
        stop.dispatchEvent(new Event("click"));
    };
    const tick = time => {
        metrics.now = time;
        const callbacks = [...frames.values()];
        frames.clear();
        callbacks.forEach(callback => callback());
    };
    return {context, sourceModes, retake: controls["[data-retake]"], inputFile, capture, record, stop, pause, open, close, cancel, root, elapsed, save, preview, status, metrics, window,
        nameVoice: controls["[data-name]"], readingTitle, done, savedDownload,
        microphone, microphoneInfo, refresh, level, levelMessage, tick, frames, track,
        prepare: () => window.vibevoiceRecorder.prepareSave([null, "My voice", "Other voice", null, null])};
}

test("closing a paused recording stops capture and reopens the retained take at Listen", async () => {
    const rig = harness();
    rig.open.dispatchEvent(new Event("click"));
    assert.equal(rig.root.open, true);
    assert.equal(rig.root.dataset.stage, "microphone");
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(rig.readingTitle.focused, true);
    rig.tick(2000);
    rig.pause.dispatchEvent(new Event("click"));
    rig.tick(6000);
    assert.equal(rig.elapsed.textContent, "0:02");
    assert.equal(rig.pause.textContent, "Resume");
    rig.close.dispatchEvent(new Event("click"));
    assert.equal(rig.metrics.stopped, 1);
    assert.equal(rig.root.open, false);
    assert.equal(rig.preview.src, "blob:fixture");
    rig.open.dispatchEvent(new Event("click"));
    assert.equal(rig.root.dataset.stage, "listen");
});

test("closing and reopening during a save keeps the result visible and a failure keeps the take", async () => {
    const rig = harness();
    rig.open.dispatchEvent(new Event("click"));
    await rig.capture();
    rig.nameVoice.dispatchEvent(new Event("click"));
    rig.save.dispatchEvent(new Event("click"));
    const upload = rig.prepare();
    rig.close.dispatchEvent(new Event("click"));
    rig.open.dispatchEvent(new Event("click"));
    assert.equal(rig.root.dataset.stage, "save");
    await upload;
    rig.window.vibevoiceRecorder.finishSave(false, true);
    assert.equal(rig.preview.src, "blob:fixture");
    assert.equal(rig.save.disabled, false);
    rig.save.dispatchEvent(new Event("click"));
    await rig.prepare();
    rig.window.vibevoiceRecorder.finishSave(true, false, {
        url: "/files/voice.wav", orig_name: "My voice.wav"
    });
    assert.equal(rig.root.dataset.stage, "save");
    assert.equal(rig.save.disabled, true);
    assert.equal(rig.preview.src, undefined);
    assert.equal(rig.savedDownload.href, "/files/voice.wav");
    assert.equal(rig.savedDownload.download, "My voice.wav");
    assert.equal(rig.done.focused, true);
    rig.done.dispatchEvent(new Event("click"));
    assert.equal(rig.root.open, false);
});

test("WAV encoder preserves rate, interleaved channels and PCM16 amplitudes", async () => {
    const rig = harness();
    const wav = rig.context.encodeVoiceWav(audio);
    const bytes = Buffer.from(await wav.arrayBuffer());
    assert.equal(bytes.toString("ascii", 0, 4), "RIFF");
    assert.equal(bytes.toString("ascii", 8, 16), "WAVEfmt ");
    assert.equal(bytes.readUInt32LE(4), bytes.length - 8);
    assert.equal(bytes.readUInt32LE(24), 44100);
    assert.equal(bytes.readUInt16LE(22), 2);
    assert.equal(bytes.readUInt16LE(34), 16);
    assert.equal(bytes.readUInt32LE(40), 12);
    assert.deepEqual(Array.from({length: 6}, (_, i) => bytes.readInt16LE(44 + i * 2)),
        [-32768, 16384, 32767, -16384, 32767, 0]);
    assert.throws(() => rig.context.encodeVoiceWav({...audio, length: 0}), /empty/);
});

test("Stop exposes local playback while preparing the WAV and only uploads on Save", async () => {
    const rig = harness();
    await rig.capture();
    assert.equal(rig.preview.src, "blob:fixture");
    assert.equal(rig.preview.hidden, false);
    assert.equal(rig.metrics.uploads, 0);
    assert.equal(rig.metrics.stopped, 1);
    assert.equal(rig.save.disabled, false);
});

test("Save encodes/uploads once per take and preserves other inputs", async () => {
    const rig = harness();
    await rig.capture();
    const result = await rig.prepare();
    assert.equal(result.length, 5);
    assert.equal(result[0].path, "/uploaded/voice.wav");
    assert.equal(result[0].meta._type, "gradio.FileData");
    assert.equal(result[1], "My voice");
    assert.equal(result[2], "Other voice");
    await rig.prepare();
    assert.equal(rig.metrics.uploads, 1);
    assert.equal(rig.metrics.decodes, 1);
    assert.equal(rig.metrics.closed, 1);
    await rig.capture();
    await rig.prepare();
    assert.equal(rig.metrics.uploads, 2);
    assert.equal(rig.metrics.decodes, 2);
    assert.equal(rig.metrics.revoked, 1);
});

test("failed upload retains preview and encoded WAV for a successful retry", async () => {
    const rig = harness();
    await rig.capture();
    const originalFetch = rig.context.fetch;
    rig.context.fetch = async () => ({ok: false});
    assert.equal((await rig.prepare())[0], null);
    assert.match(rig.status.textContent, /Upload failed/);
    assert.equal(rig.preview.src, "blob:fixture");
    assert.equal(rig.save.disabled, false);
    rig.context.fetch = originalFetch;
    assert.equal((await rig.prepare())[0].path, "/uploaded/voice.wav");
    assert.equal(rig.metrics.decodes, 1);
});

test("hung upload times out, aborts, restores controls and allows retry", async () => {
    const rig = harness();
    await rig.capture();
    const originalFetch = rig.context.fetch;
    let signal;
    rig.context.fetch = async (_url, options) => {
        signal = options.signal;
        return await new Promise(() => {});
    };
    const pending = rig.prepare();
    assert.equal(rig.save.disabled, true);
    const secondClick = new Event("click", {cancelable: true});
    rig.save.dispatchEvent(secondClick);
    assert.equal(secondClick.defaultPrevented, true);
    assert.equal((await pending)[0], null);
    assert.equal(signal.aborted, true);
    assert.match(rig.status.textContent, /took too long/);
    assert.equal(rig.save.disabled, false);
    assert.equal(rig.preview.src, "blob:fixture");
    rig.context.fetch = originalFetch;
    assert.equal((await rig.prepare())[0].path, "/uploaded/voice.wav");
});

test("microphone permission failure preserves the previous take", async () => {
    const rig = harness();
    await rig.capture();
    rig.context.navigator.mediaDevices.getUserMedia = async () => { throw new Error("denied"); };
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    assert.match(rig.status.textContent, /Check your microphone/);
    assert.equal(rig.record.disabled, false);
    assert.equal((await rig.prepare())[0].path, "/uploaded/voice.wav");
});

test("hung decoder is closed on timeout and a fresh decode can succeed", async () => {
    const rig = harness();
    const originalContext = rig.context.AudioContext;
    let lateDecode;
    rig.context.AudioContext = class extends originalContext {
        async decodeAudioData() { return new Promise(resolve => { lateDecode = resolve; }); }
    };
    await rig.capture();
    assert.equal((await rig.prepare())[0], null);
    assert.equal(rig.metrics.closed, 1);
    rig.context.AudioContext = originalContext;
    assert.equal((await rig.prepare())[0].path, "/uploaded/voice.wav");
    lateDecode({...audio, length: 0});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal((await rig.prepare())[0].path, "/uploaded/voice.wav");
    assert.equal(rig.metrics.decodes, 1);
    assert.equal(rig.metrics.uploads, 1);
});

test("restoring a cached browser page restores preview and recording on repeated visits", async () => {
    const rig = harness();
    await rig.capture();
    for (let visit = 0; visit < 2; visit++) {
        rig.window.dispatchEvent(new Event("pagehide"));
        const show = new Event("pageshow");
        show.persisted = true;
        rig.window.dispatchEvent(show);
        assert.equal(rig.preview.src, "blob:fixture");
        await rig.capture();
        assert.equal(rig.stop.disabled, true);
        assert.match(rig.status.textContent, /Ready to play/);
    }
    assert.equal(rig.metrics.stopped, 3);
    assert.equal(rig.metrics.revoked, 4);
});

test("microphone choices list inputs and request the chosen device exactly", async () => {
    const rig = harness();
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(rig.microphone.options.map(option => option.value), ["", "virtual", "wireless"]);
    rig.microphone.value = "wireless";
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(rig.metrics.constraints.audio.deviceId.exact, "wireless");
    assert.match(rig.microphoneInfo.textContent, /Recording from Example USB Microphone/);
    assert.equal(rig.microphone.disabled, true);
    rig.stop.dispatchEvent(new Event("click"));
    assert.equal(rig.microphone.disabled, false);
});

test("live level responds to sound and silence, then stops updating with recording", async () => {
    const rig = harness();
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    rig.metrics.signal = 0.05;
    rig.tick(100);
    assert.ok(rig.level.value > 50);
    assert.equal(rig.levelMessage.textContent, "Sound detected.");
    rig.metrics.signal = 0;
    rig.tick(3000);
    assert.equal(rig.level.value, 0);
    assert.match(rig.levelMessage.textContent, /No input detected/);
    rig.metrics.signal = 0.03;
    rig.tick(3100);
    assert.equal(rig.levelMessage.textContent, "Sound detected.");
    rig.stop.dispatchEvent(new Event("click"));
    assert.equal(rig.frames.size, 0);
    assert.equal(rig.metrics.meterClosed, 1);
    assert.equal(rig.metrics.disconnected, 1);
    assert.equal(rig.levelMessage.textContent, "Microphone stopped.");
});

test("missing selected microphone reports the device error and preserves previous take", async () => {
    const rig = harness();
    await rig.capture();
    rig.microphone.value = "missing";
    let requested;
    rig.context.navigator.mediaDevices.getUserMedia = async constraints => {
        requested = constraints;
        throw Object.assign(new Error("missing"), {name: "OverconstrainedError"});
    };
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requested.audio.deviceId.exact, "missing");
    assert.match(rig.status.textContent, /selected microphone is unavailable/);
    assert.equal(rig.preview.src, "blob:fixture");
    assert.equal(rig.save.disabled, false);
});

test("cancelled permission request releases late microphone access and preserves previous take", async () => {
    const rig = harness();
    await rig.capture();
    let grant;
    rig.context.navigator.mediaDevices.getUserMedia = async () => await new Promise(resolve => { grant = resolve; });
    rig.record.dispatchEvent(new Event("click"));
    assert.equal(rig.cancel.hidden, false);
    rig.cancel.dispatchEvent(new Event("click"));
    assert.equal(rig.record.disabled, false);
    let released = false;
    grant({getTracks: () => [{stop: () => { released = true; }}]});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(released, true);
    assert.equal(rig.preview.src, "blob:fixture");
    assert.match(rig.status.textContent, /request cancelled/);
});

test("a suspended level meter reports its state without claiming input is silent", async () => {
    const rig = harness();
    const Native = rig.context.AudioContext;
    rig.context.AudioContext = class extends Native {
        constructor(...args) { super(...args); this.state = "suspended"; }
        async resume() {}
    };
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    rig.tick(4000);
    assert.match(rig.levelMessage.textContent, /meter paused/);
    assert.equal(rig.stop.disabled, false);
    rig.stop.dispatchEvent(new Event("click"));
    assert.equal(rig.preview.src, "blob:fixture");
});

test("microphone disconnect stops capture and meter while retaining an earlier take", async () => {
    const rig = harness();
    await rig.capture();
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    rig.track.dispatchEvent(new Event("ended"));
    assert.match(rig.status.textContent, /microphone disconnected/);
    assert.equal(rig.frames.size, 0);
    assert.equal(rig.record.disabled, false);
    assert.equal(rig.preview.src, "blob:fixture");
});

test("late access from a cancelled request cannot interrupt a newer recording", async () => {
    const rig = harness();
    const originalAccess = rig.context.navigator.mediaDevices.getUserMedia;
    let oldGrant;
    rig.context.navigator.mediaDevices.getUserMedia = async () => await new Promise(resolve => { oldGrant = resolve; });
    rig.record.dispatchEvent(new Event("click"));
    rig.cancel.dispatchEvent(new Event("click"));
    rig.context.navigator.mediaDevices.getUserMedia = originalAccess;
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    let released = false;
    oldGrant({getTracks: () => [{stop: () => { released = true; }}]});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(released, true);
    assert.equal(rig.stop.disabled, false);
    assert.equal(rig.record.disabled, true);
    assert.equal(rig.frames.size, 1);
    assert.equal(rig.root.dataset.stage, "record");
    rig.stop.dispatchEvent(new Event("click"));
    assert.equal(rig.frames.size, 0);
});

test("a failed level meter leaves recording and playback working", async () => {
    const rig = harness();
    const Native = rig.context.AudioContext;
    rig.context.AudioContext = class extends Native {
        createAnalyser() { throw new Error("meter unavailable"); }
    };
    rig.record.dispatchEvent(new Event("click"));
    await new Promise(resolve => setImmediate(resolve));
    assert.match(rig.levelMessage.textContent, /level meter is unavailable/);
    assert.equal(rig.stop.disabled, false);
    rig.stop.dispatchEvent(new Event("click"));
    assert.equal(rig.preview.src, "blob:fixture");
    assert.equal(rig.metrics.meterClosed, 1);
});


test("hung import decoder closes and allows a new WAV import", async () => {
    const rig = harness();
    const wav = vm.runInContext("encodeVoiceWav", rig.context)(audio);
    rig.inputFile.files = [wav];
    const Context = rig.context.AudioContext;
    rig.context.AudioContext = class extends Context {
        decodeAudioData() { return new Promise(() => {}); }
    };
    rig.inputFile.dispatchEvent(new Event("change"));
    await new Promise(resolve => setTimeout(resolve, 30));
    assert.equal(rig.metrics.closed, 1);
    rig.context.AudioContext = Context;
    rig.inputFile.dispatchEvent(new Event("change"));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(rig.metrics.decodes, 1);
    assert.equal(rig.metrics.closed, 2);
});


test("source choices stay out of recording listen and save then return at initial step", async () => {
    const rig = harness();
    rig.open.dispatchEvent(new Event("click"));
    assert.equal(rig.sourceModes.hidden, false);
    rig.record.dispatchEvent(new Event("click"));
    assert.equal(rig.sourceModes.hidden, true);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(rig.root.dataset.stage, "record");
    assert.equal(rig.sourceModes.hidden, true);
    rig.stop.dispatchEvent(new Event("click"));
    assert.equal(rig.root.dataset.stage, "listen");
    assert.equal(rig.sourceModes.hidden, true);
    rig.close.dispatchEvent(new Event("click"));
    rig.open.dispatchEvent(new Event("click"));
    assert.equal(rig.sourceModes.hidden, true);
    rig.nameVoice.dispatchEvent(new Event("click"));
    assert.equal(rig.sourceModes.hidden, true);
    rig.retake.dispatchEvent(new Event("click"));
    assert.equal(rig.root.dataset.stage, "microphone");
    assert.equal(rig.sourceModes.hidden, false);
});
