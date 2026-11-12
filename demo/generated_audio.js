async function readGeneratedWaveform(url, signal) {
    const response = await fetch(url, { mode: "same-origin", signal });
    if (!response.ok || !response.body) return [];
    const reader = response.body.getReader();
    const peaks = new Float32Array(96);
    let header = new Uint8Array(0);
    let total = 0;
    let received = 0;
    let sampleByte = 1;
    let stride = 2;

    try {
        while (!total || received < total) {
            const { done, value } = await reader.read();
            if (done) break;
            let chunk = value;
            if (!total) {
                const bytes = new Uint8Array(header.length + chunk.length);
                bytes.set(header);
                bytes.set(chunk, header.length);
                header = bytes;
                if (header.length < 12) continue;
                const view = new DataView(header.buffer);
                if (view.getUint32(0) !== 0x52494646 || view.getUint32(8) !== 0x57415645) return [];
                let frameBytes = 0;
                for (let offset = 12; offset + 8 <= header.length;) {
                    const name = view.getUint32(offset);
                    const length = view.getUint32(offset + 4, true);
                    const start = offset + 8;
                    if (name === 0x64617461) {
                        if (!frameBytes || length < frameBytes || length % frameBytes) return [];
                        total = length;
                        // Sample the full take without retaining or decoding the full recording.
                        stride = frameBytes * Math.max(1, Math.floor(total / frameBytes / 500000));
                        chunk = header.subarray(start);
                        header = null;
                        break;
                    }
                    if (start + length > header.length) break;
                    if (name === 0x666d7420 && length >= 16) {
                        const channels = view.getUint16(start + 2, true);
                        if (view.getUint16(start, true) !== 1 || view.getUint16(start + 14, true) !== 16) return [];
                        frameBytes = channels * 2;
                    }
                    offset = start + length + (length % 2);
                }
                if (!total) {
                    if (header.length > 65536) return [];
                    continue;
                }
            }
            const end = Math.min(total, received + chunk.length);
            for (; sampleByte < end; sampleByte += stride) {
                const sample = chunk[sampleByte - received];
                const amplitude = Math.abs(sample < 128 ? sample : sample - 256) / 128;
                const bin = Math.floor(sampleByte * peaks.length / total);
                peaks[bin] = Math.max(peaks[bin], amplitude);
            }
            received = end;
        }
        return total && received === total ? peaks : [];
    } finally {
        await reader.cancel().catch(() => {});
    }
}

function initGeneratedAudio() {
    const composer = document.getElementById("guided-composer");
    const liveRegion = document.getElementById("live-generated-audio");
    const completed = document.getElementById("generated-audio-player");
    const audio = completed.querySelector("audio");
    const play = completed.querySelector("#playback-toggle");
    const wave = completed.querySelector("#playback-waveform");
    const seek = completed.querySelector("#playback-seek");
    const title = completed.querySelector("#playback-title");
    const elapsed = completed.querySelector("#playback-elapsed");
    const duration = completed.querySelector("#playback-duration");
    const download = completed.querySelector("#download-wav");
    const replay = completed.querySelector("[data-replay-audio]");
    const clear = completed.querySelector("[data-clear-audio]");
    const attempts = new WeakSet();
    let liveAudio;
    let previousAudio;
    let previousSource;
    let requested = false;
    let generating = false;
    let paused = false;
    let cancelled = false;
    let lastFileUrl;
    let waveform = [];
    let waveformRequest;
    let liveTransfer = null;
    let awaitingOutcome = false;
    let completionVersion = 0;

    const formatTime = seconds => {
        const whole = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
        return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
    };

    function silence(player) {
        if (player) {
            player.muted = true;
            player.pause();
        }
    }

    function drawWaveform(progress) {
        const context = wave.getContext("2d");
        const width = wave.width;
        const height = wave.height;
        if (!context || !width || !height) return;
        context.clearRect(0, 0, width, height);
        const peaks = generating ? [] : waveform;
        const max = peaks.length ? Math.max(...peaks) : 0;
        if (!max) {
            context.fillStyle = "#82b5a2";
            context.fillRect(0, height / 2, width, 1);
            return;
        }
        const spacing = width / peaks.length;
        for (let i = 0; i < peaks.length; i++) {
            const bar = Math.max(2, peaks[i] / max * height * 0.86);
            context.fillStyle = i / peaks.length <= progress ? "#e2fff1" : "#82b5a2";
            context.fillRect(i * spacing, (height - bar) / 2, Math.max(1, spacing * 0.58), bar);
        }
    }

    function currentLivePlayer() {
        const source = liveAudio?.getAttribute("src") || liveAudio?.currentSrc;
        return liveAudio && source && (liveAudio !== previousAudio || source !== previousSource) ? liveAudio : null;
    }

    function publishPending() {
        const generate = document.getElementById("generate-audio");
        if (generate && awaitingOutcome) generate.disabled = true;
        window.dispatchEvent(new Event("vibevoice-generation-state"));
    }

    // Protect the server-finished / final-file callback interval too. Script edits
    // must not enqueue a new run before its predecessor has handed off playback.
    document.addEventListener("click", event => {
        if (awaitingOutcome && event.target.closest?.("#generate-audio, button[data-section-command]")) {
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }, {capture: true});

    function rememberLivePosition() {
        const player = currentLivePlayer();
        if ((requested || generating) && player && Number.isFinite(player.currentTime)) {
            liveTransfer = {position: player.currentTime, playing: !paused && player.readyState >= 2};
        }
    }

    function render() {
        if (generating) rememberLivePosition();
        const player = generating ? currentLivePlayer() : audio;
        const hasTake = audio.hasAttribute("src");
        const total = !generating && hasTake && Number.isFinite(audio.duration) ? audio.duration : 0;
        const current = player?.currentTime || 0;
        const progress = total ? Math.min(1, current / total) : 0;
        const label = player && !player.paused && !player.ended ? "Pause" : "Play";
        if (play.textContent !== label) play.textContent = label;
        play.disabled = generating ? !player || player.readyState < 2 : !hasTake || audio.readyState < 1 || (awaitingOutcome && !cancelled);
        seek.disabled = !total;
        seek.value = Math.round(progress * 1000);
        seek.setAttribute("aria-valuetext", formatTime(current) + " of " + formatTime(total));
        elapsed.textContent = formatTime(current);
        duration.textContent = generating ? "Live" : formatTime(total);
        title.textContent = generating
            ? (player && player.readyState >= 2 ? "Live audio" : "Generating audio…")
            : awaitingOutcome && !cancelled ? "Finishing audio." : hasTake ? "Last completed take" : "No audio yet";
        download.title = generating && hasTake ? "Download last completed take" : "";
        clear.disabled = generating || (awaitingOutcome && !cancelled) || !hasTake;
        replay.disabled = clear.disabled || audio.readyState < 1;
        drawWaveform(progress);
    }

    function finishLive() {
        rememberLivePosition();
        requested = generating = false;
        silence(liveAudio);
        render();
    }

    function syncLive() {
        const next = liveRegion.querySelector("audio");
        if (next !== liveAudio) {
            rememberLivePosition();
            silence(liveAudio);
            liveAudio = next;
        }
        const stop = composer.querySelector("#stop-generation");
        const showingStop = Boolean(stop?.getClientRects().length);
        if (generating && !showingStop) {
            finishLive();
            return;
        }
        generating = requested && showingStop;
        const player = generating && !paused ? currentLivePlayer() : null;
        if (!player) {
            silence(liveAudio);
            render();
            return;
        }
        player.muted = false;
        if (player.readyState >= 2 && player.paused && !player.ended && !attempts.has(player)) {
            attempts.add(player);
            player.play().catch(error => {
                if (error.name !== "AbortError" && player === liveAudio && generating) paused = true;
            }).finally(() => {
                attempts.delete(player);
                if (!generating || paused || player !== currentLivePlayer()) silence(player);
                render();
            });
        }
        render();
    }

    function setDownload(url, name) {
        if (url) {
            download.href = url;
            download.download = name || "generated-audio.wav";
        } else {
            download.removeAttribute("href");
            download.removeAttribute("download");
        }
        download.setAttribute("aria-disabled", String(!url));
        download.tabIndex = url ? 0 : -1;
    }

    play.addEventListener("click", () => {
        if (play.disabled) return;
        if (generating) {
            const player = currentLivePlayer();
            if (player) {
                paused = !player.paused && !player.muted;
                syncLive();
            }
        } else if (audio.paused || audio.ended) {
            audio.play().catch(render);
        } else {
            audio.pause();
        }
    });
    seek.addEventListener("input", () => {
        if (!generating && !seek.disabled) {
            audio.currentTime = Number(seek.value) * audio.duration / 1000;
            render();
        }
    });
    replay.addEventListener("click", () => {
        if (replay.disabled) return;
        audio.currentTime = 0;
        audio.play().catch(render);
    });
    clear.addEventListener("click", () => {
        if (clear.disabled) return;
        completionVersion++;
        liveTransfer = null;
        waveformRequest?.abort();
        waveform = [];
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
        setDownload();
        render();
        composer.querySelector("#generate-audio")?.focus();
    });

    // Gradio can replace the media element while delivering an audio chunk.
    const observer = new MutationObserver(syncLive);
    observer.observe(liveRegion, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
    observer.observe(document.getElementById("composer-actions"), {
        childList: true, subtree: true, attributes: true, attributeFilter: ["style", "hidden", "class"]
    });
    liveRegion.addEventListener("canplay", syncLive, true);
    liveRegion.addEventListener("play", event => {
        if (!generating || paused || event.target !== currentLivePlayer()) silence(event.target);
        render();
    }, true);
    for (const event of ["loadedmetadata", "durationchange", "timeupdate", "pause", "ended", "error"]) {
        audio.addEventListener(event, render);
        liveRegion.addEventListener(event, render, true);
    }
    audio.addEventListener("play", render);
    window.addEventListener("pagehide", () => {
        completionVersion++;
        finishLive();
        audio.pause();
    });

    window.vibevoicePlayback = {
        isPending() { return awaitingOutcome; },
        start() {
            finishLive();
            completionVersion++;
            liveTransfer = null;
            audio.pause();
            previousAudio = liveRegion.querySelector("audio");
            previousSource = previousAudio?.getAttribute("src") || previousAudio?.currentSrc;
            silence(previousAudio);
            requested = true;
            awaitingOutcome = true;
            publishPending();
            cancelled = false;
            paused = false;
        },
        stop() {
            cancelled = true;
            // Cancellation ends listening immediately, but submission stays locked
            // until the server reports that the worker has actually finished.
            publishPending();
            completionVersion++;
            finishLive();
            liveTransfer = null;
        },
        acceptExport() {
            finishLive();
            awaitingOutcome = true;
            publishPending();
            completionVersion++;
            liveTransfer = null;
            cancelled = false;
            requested = false;
        },
        finish(outcome, file) {
            if (!outcome || outcome.running) return;
            if (outcome.completed && !cancelled && this.complete(file)) return;
            finishLive();
            liveTransfer = null;
            awaitingOutcome = false;
            publishPending();
            render();
        },
        complete(file) {
            if (!file?.url || cancelled) return false;
            let url;
            try {
                url = new URL(file.url, window.location.href);
            } catch {
                return false;
            }
            if (url.origin !== window.location.origin) return false;
            if (url.href === lastFileUrl) {
                finishLive();
                liveTransfer = null;
                awaitingOutcome = false;
                publishPending();
                render();
                return true;
            }
            finishLive();
            awaitingOutcome = false;
            publishPending();
            const transfer = liveTransfer;
            liveTransfer = null;
            const version = ++completionVersion;
            lastFileUrl = url.href;
            waveformRequest?.abort();
            const request = new AbortController();
            waveformRequest = request;
            waveform = [];
            audio.pause();
            audio.src = url.href;
            const resumeCompleted = () => {
                audio.removeEventListener("loadedmetadata", resumeCompleted);
                if (version !== completionVersion || cancelled) return;
                if (transfer) {
                    const total = Number.isFinite(audio.duration) ? audio.duration : transfer.position;
                    audio.currentTime = Math.max(0, Math.min(transfer.position, total));
                    // HTMLMediaElement.play() at the end restarts from zero.
                    // An exhausted take stays finished until the user explicitly plays again.
                    if (transfer.playing && audio.currentTime < total) audio.play().then(() => {
                        if ((version !== completionVersion || cancelled) && audio.src === url.href) audio.pause();
                        render();
                    }).catch(render);
                }
                render();
            };
            audio.addEventListener("loadedmetadata", resumeCompleted, {once: true});
            if (audio.readyState >= 1) resumeCompleted();
            setDownload(url.href, file.orig_name);
            render();
            readGeneratedWaveform(url.href, request.signal).then(peaks => {
                if (!request.signal.aborted) {
                    waveform = peaks;
                    render();
                }
            }).catch(() => {});
            return true;
        },
    };
    setDownload();
    render();
}
