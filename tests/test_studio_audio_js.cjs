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

test('studio exposes modal Stop and resets export acceptance', () => {
    const studio = fs.readFileSync(path.join(__dirname, '../demo/studio_audio.js'), 'utf8');
    assert.match(studio, /data-stop-studio/);
    assert.match(studio, /stopHere\.disabled\) stop\.click\(\)/);
    assert.match(studio, /endsWith\(":export"\)\) window\.vibevoicePlayback\?\.acceptExport\(\)/);
});
