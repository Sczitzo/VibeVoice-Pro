const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../demo/generated_audio.js'), 'utf8');
const methods = source.slice(source.indexOf('    window.vibevoicePlayback = {'), source.lastIndexOf('    };') + 6);

test('explicit section export after Stop accepts a completed file without enabling live audio', () => {
    const audio = {pause() {}, addEventListener() {}, removeEventListener() {}};
    const sandbox = {
        window: {location: {href: 'http://localhost:7881/', origin: 'http://localhost:7881'}},
        awaitingOutcome: false, completionVersion: 0, liveTransfer: null, URL, AbortController, audio, cancelled: false, requested: false, paused: false,
        publishPending() {}, finishLive() {}, render() {}, setDownload() {}, silence() {},
        liveRegion: {querySelector() {return null;}},
        lastFileUrl: null, waveformRequest: null, waveform: [],
        readGeneratedWaveform: () => Promise.resolve([])
    };
    vm.runInNewContext(methods, sandbox);
    const playback = sandbox.window.vibevoicePlayback;
    playback.stop();
    playback.complete({url: '/old.wav'});
    assert.equal(audio.src, undefined, 'late stopped output remains ignored');
    playback.acceptExport();
    assert.equal(sandbox.requested, false, 'export is not a live generation request');
    playback.complete({url: '/export.wav'});
    assert.equal(audio.src, 'http://localhost:7881/export.wav');
});

test('section retry closes the library so Stop is reachable in the composer', () => {
    const studio = fs.readFileSync(path.join(__dirname, '../demo/studio_audio.js'), 'utf8');
    const source = studio.slice(studio.indexOf('        const action = '), studio.indexOf('        const preview = '));
    const calls = [];
    const field = {dispatchEvent() {}};
    const action = {dataset: {sectionCommand: 'job:0'}};
    const sandbox = {
        event: {target: {closest: () => action}},
        document: {querySelector: () => field, getElementById: () => ({click: () => calls.push('submit')})},
        dialog: {close: () => calls.push('close')},
        window: {vibevoicePlayback: {start: () => calls.push('start'), acceptExport: () => calls.push('export')}},
        requestAnimationFrame: fn => fn(), Event,
    };
    vm.runInNewContext('(() => {' + source + '})()', sandbox);
    assert.deepEqual(calls.splice(0), ['start', 'close', 'submit']);
    action.dataset.sectionCommand = 'job:export';
    vm.runInNewContext('(() => {' + source + '})()', sandbox);
    assert.deepEqual(calls, ['export', 'submit']);
});
