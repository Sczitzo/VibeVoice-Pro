function encodeVoiceWav(audio) {
    const channels = audio.numberOfChannels;
    const dataSize = audio.length * channels * 2;
    if (!channels || !audio.length || dataSize > 0xffffffff - 36) {
        throw new Error("The recording is empty or too large to save as WAV.");
    }
    const bytes = new ArrayBuffer(44 + dataSize);
    const view = new DataView(bytes);
    const writeText = (offset, text) => {
        for (let index = 0; index < text.length; index++) {
            view.setUint8(offset + index, text.charCodeAt(index));
        }
    };
    writeText(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeText(8, "WAVEfmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, audio.sampleRate, true);
    view.setUint32(28, audio.sampleRate * channels * 2, true);
    view.setUint16(32, channels * 2, true);
    view.setUint16(34, 16, true);
    writeText(36, "data");
    view.setUint32(40, dataSize, true);
    const samples = Array.from({length: channels}, (_, channel) => audio.getChannelData(channel));
    let offset = 44;
    for (let frame = 0; frame < audio.length; frame++) {
        for (const channel of samples) {
            const sample = Math.max(-1, Math.min(1, channel[frame]));
            view.setInt16(offset, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
            offset += 2;
        }
    }
    return new Blob([bytes], {type: "audio/wav"});
}

function initVoiceRecorder() {
    const root = document.getElementById("voice-recorder");
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "true";
    const record = root.querySelector("[data-record]");
    const stop = root.querySelector("[data-stop]");
    const pause = root.querySelector("[data-pause]");
    const openDialog = document.getElementById("open-voice-recorder");
    const closeDialog = root.querySelector("[data-close]");
    const done = root.querySelector("[data-done]");
    const savedDownload = root.querySelector("[data-saved-download]");
    const cancel = root.querySelector("[data-cancel]");
    const retake = root.querySelector("[data-retake]");
    const returnToTake = root.querySelector("[data-return]");
    const nameVoice = root.querySelector("[data-name]");
    const back = root.querySelector("[data-back]");
    const elapsed = root.querySelector("[data-elapsed]");
    const readingTitle = root.querySelector("#voice-reading-title");
    const captureState = root.querySelector("[data-capture-state]");
    const activeDevice = root.querySelector("[data-active-device]");
    const bars = [...root.querySelectorAll(".recorder-bars i")];
    const panes = root.querySelectorAll("[data-pane]");
    const steps = root.querySelectorAll("[data-step]");
    const microphone = root.querySelector("[data-microphone]");
    const refresh = root.querySelector("[data-refresh]");
    const microphoneInfo = root.querySelector("[data-microphone-info]");
    const level = root.querySelector("[data-level]");
    const levelMessage = root.querySelector("[data-level-message]");
    const preview = root.querySelector("audio");
    const status = root.querySelector("[data-recording-status]");
    const save = document.getElementById("save-voice-btn");
    const savePanel = document.getElementById("voice-save-panel");
    root.querySelector("[data-save-controls]").append(savePanel);
    save.classList.add("primary");
    const voiceName = savePanel.querySelector("#voice-name input, #voice-name textarea");
    const saveResult = savePanel.querySelector("#voice-save-status");
    saveResult.setAttribute("role", "status");
    voiceName.autocomplete = "off";
    voiceName.spellcheck = false;
    const script = document.querySelector("#composer-editor textarea");
    script.setAttribute("aria-labelledby", "composer-script-label");
    script.setAttribute("aria-describedby", "composer-script-help");
    let recorder = null;
    let clip = null;
    let previewUrl = null;
    let busy = false;
    let disposed = false;
    let permissionRequest = 0;
    let waitingPermission = false;
    let stopMeter = () => {};
    let elapsedMs = 0;
    let startedAt = null;
    let clock = null;
    let savePending = false;

    function setStage(stage) {
        root.dataset.stage = stage;
        for (const pane of panes) pane.hidden = pane.dataset.pane !== stage;
        for (const step of steps) {
            if (step.dataset.step === stage) step.setAttribute("aria-current", "step");
            else step.removeAttribute("aria-current");
        }
        updateControls();
        if (root.open) {
            const control = {microphone, record: readingTitle, listen: preview, save: voiceName}[stage];
            control.focus();
        }
    }

    function showStatus(message, error = false) {
        status.textContent = message;
        status.toggleAttribute("data-error", error);
    }

    function updateElapsed() {
        const seconds = Math.floor((elapsedMs + (startedAt === null ? 0 : performance.now() - startedAt)) / 1000);
        elapsed.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    }

    function holdClock() {
        if (startedAt !== null) elapsedMs += performance.now() - startedAt;
        startedAt = null;
        updateElapsed();
    }

    function updateLevel(value) {
        level.value = value;
        bars.forEach((bar, index) => {
            const weight = 1 - Math.abs(index - 4) * 0.18;
            bar.style.setProperty("--bar-scale", (0.12 + 0.88 * value / 100 * weight).toFixed(3));
        });
    }

    async function refreshMicrophones() {
        if (!navigator.mediaDevices?.enumerateDevices) {
            microphoneInfo.textContent = "Microphone access is unavailable. Open this app at http://127.0.0.1:7860 in your browser.";
            return;
        }
        try {
            const devices = (await navigator.mediaDevices.enumerateDevices())
                .filter(device => device.kind === "audioinput" && device.deviceId && device.deviceId !== "default");
            const selected = microphone.value;
            const options = devices.map((device, index) => new Option(device.label || `Microphone ${index + 1}`, device.deviceId));
            if (selected && !devices.some(device => device.deviceId === selected)) {
                options.push(new Option("Selected microphone is unavailable. Choose another.", selected));
            }
            microphone.replaceChildren(new Option("System default microphone", ""), ...options);
            microphone.value = selected;
            if (!recorder) {
                microphoneInfo.textContent = devices.some(device => device.label)
                    ? "Choose your microphone, then start recording."
                    : "Start recording to allow access and reveal microphone names.";
            }
        } catch (error) {
            microphoneInfo.textContent = "Could not list microphones. Check browser permission, then click Refresh microphones.";
        }
    }

    function monitorMicrophone(stream) {
        let context;
        let source;
        let animation;
        let closed = false;
        const close = () => {
            if (closed) return;
            closed = true;
            if (animation !== undefined) cancelAnimationFrame(animation);
            if (source) source.disconnect();
            if (context) context.close().catch(() => {});
            updateLevel(0);
        };
        try {
            context = new AudioContext();
            source = context.createMediaStreamSource(stream);
            const analyser = context.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);
            const samples = new Float32Array(analyser.fftSize);
            let lastSound = performance.now();
            const draw = () => {
                if (closed) return;
                if (recorder?.state === "paused") {
                    updateLevel(0);
                } else if (context.state !== "running") {
                    updateLevel(0);
                    levelMessage.textContent = "Level meter paused. Recording can continue. Stop and try again to check the input.";
                } else {
                    analyser.getFloatTimeDomainData(samples);
                    let energy = 0;
                    for (const sample of samples) energy += sample * sample;
                    const rms = Math.sqrt(energy / samples.length);
                    updateLevel(Math.max(0, Math.min(100, (20 * Math.log10(Math.max(rms, 0.000001)) + 60) * 100 / 60)));
                    if (rms >= 0.003) lastSound = performance.now();
                    const message = rms >= 0.003 ? "Sound detected."
                        : performance.now() - lastSound >= 2500
                            ? "No input detected. Check the selected microphone, its mute switch, and its volume."
                            : "Speak into your microphone and watch the level meter.";
                    if (levelMessage.textContent !== message) levelMessage.textContent = message;
                }
                animation = requestAnimationFrame(draw);
            };
            context.resume().then(draw).catch(() => {
                if (!closed) levelMessage.textContent = "The level meter could not start. Recording can continue.";
                close();
            });
        } catch (error) {
            levelMessage.textContent = "The level meter is unavailable. Record a short take and use Play to check it.";
            close();
        }
        return close;
    }

    function updateControls() {
        const locked = busy || savePending;
        record.disabled = locked || recorder !== null;
        stop.disabled = busy || recorder === null;
        pause.disabled = stop.disabled;
        pause.textContent = recorder?.state === "paused" ? "Resume" : "Pause";
        cancel.hidden = !waitingPermission;
        returnToTake.hidden = !clip || waitingPermission;
        returnToTake.disabled = locked;
        microphone.disabled = record.disabled;
        refresh.disabled = record.disabled;
        save.disabled = locked || recorder !== null || !clip;
        back.disabled = locked || !clip;
    }

    openDialog.addEventListener("click", () => {
        if (root.open) return;
        if (!recorder && !waitingPermission) {
            if (savePending) setStage("save");
            else setStage(clip ? "listen" : "microphone");
        }
        root.showModal();
        if (root.dataset.stage === "microphone") microphone.focus();
        if (root.dataset.stage === "listen") preview.focus();
        if (root.dataset.stage === "save") voiceName.focus();
        refreshMicrophones();
    });

    function dismissRecorder() {
        if (waitingPermission) cancelPermission();
        if (recorder && recorder.state !== "inactive" && !busy) stopRecording();
        preview.pause();
        root.close();
        openDialog.focus();
    }

    closeDialog.addEventListener("click", dismissRecorder);
    done.addEventListener("click", dismissRecorder);
    root.addEventListener("cancel", event => {
        event.preventDefault();
        dismissRecorder();
    });
    retake.addEventListener("click", () => {
        showStatus("");
        preview.pause();
        setStage("microphone");
    });
    returnToTake.addEventListener("click", () => setStage("listen"));
    nameVoice.addEventListener("click", () => {
        root.removeAttribute("data-saved");
        saveResult.hidden = true;
        preview.pause();
        showStatus("");
        setStage("save");
    });
    back.addEventListener("click", () => setStage("listen"));
    pause.addEventListener("click", () => {
        if (!recorder || busy) return;
        if (recorder.state === "recording") {
            recorder.pause();
            holdClock();
            captureState.textContent = "Paused";
            levelMessage.textContent = "Resume to continue.";
            updateLevel(0);
        } else if (recorder.state === "paused") {
            recorder.resume();
            startedAt = performance.now();
            captureState.textContent = "Recording";
            levelMessage.textContent = "Listening for sound…";
        }
        updateControls();
    });

    record.addEventListener("click", async () => {
        if (busy || savePending || recorder) return;
        const request = ++permissionRequest;
        waitingPermission = true;
        busy = true;
        updateControls();
        showStatus("Waiting for microphone access. Choose Allow if your browser asks.");
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: microphone.value ? {deviceId: {exact: microphone.value}} : true
            });
            if (disposed || request !== permissionRequest) {
                stream.getTracks().forEach(track => track.stop());
                return;
            }
            const active = new MediaRecorder(stream);
            let releaseMeter = () => {};
            const chunks = [];
            const track = stream.getAudioTracks()[0];
            const sampleRate = track.getSettings().sampleRate;
            microphoneInfo.textContent = `Recording from ${track.label || "the selected microphone"}.`;
            activeDevice.textContent = track.label || "Selected microphone";
            let failed = false;
            active.addEventListener("dataavailable", event => {
                if (event.data.size) chunks.push(event.data);
            });
            active.addEventListener("error", () => {
                failed = true;
                releaseMeter();
                showStatus("Microphone recording failed. Check your microphone and try again.", true);
            });
            track.addEventListener("ended", () => {
                if (active.state === "inactive") return;
                failed = true;
                releaseMeter();
                showStatus("The microphone disconnected. Reconnect it, refresh the list, and record again.", true);
                active.stop();
            });
            active.addEventListener("stop", () => {
                clearInterval(clock);
                holdClock();
                releaseMeter();
                levelMessage.textContent = "Microphone stopped.";
                stream.getTracks().forEach(track => track.stop());
                recorder = null;
                busy = false;
                if (!failed && !disposed && request === permissionRequest) {
                    const blob = new Blob(chunks, {type: active.mimeType});
                    if (blob.size) {
                        const take = {blob, sampleRate, wav: null, uploaded: null, preparing: null};
                        clip = take;
                        preview.pause();
                        if (previewUrl) URL.revokeObjectURL(previewUrl);
                        previewUrl = URL.createObjectURL(blob);
                        preview.src = previewUrl;
                        preview.hidden = false;
                        voiceName.value = "";
                        voiceName.dispatchEvent(new Event("input", {bubbles: true}));
                        setStage("listen");
                        const preparingMessage = "Ready to play. Preparing WAV…";
                        showStatus(preparingMessage);
                        prepareWav(take).then(() => {
                            if (take === clip && status.textContent === preparingMessage) showStatus("Ready to play.");
                        }).catch(() => {
                            if (take === clip && status.textContent === preparingMessage) {
                                showStatus("You can play your take. Save Voice will retry preparing the WAV.");
                            }
                        });
                    } else {
                        setStage(clip ? "listen" : "microphone");
                        showStatus("The recording was empty. Try recording again.", true);
                    }
                } else if (!disposed) {
                    setStage(clip ? "listen" : "microphone");
                }
                updateControls();
            });
            active.start();
            recorder = active;
            elapsedMs = 0;
            startedAt = performance.now();
            updateElapsed();
            clock = setInterval(updateElapsed, 250);
            captureState.textContent = "Recording";
            levelMessage.textContent = "Listening for sound…";
            releaseMeter = monitorMicrophone(stream);
            stopMeter = releaseMeter;
            refreshMicrophones();
            showStatus("");
        } catch (error) {
            if (stream) stream.getTracks().forEach(track => track.stop());
            if (disposed || request !== permissionRequest) return;
            const messages = {
                NotAllowedError: "Microphone access is blocked. Allow microphone access for this page in your browser and in Windows microphone privacy settings.",
                NotFoundError: "No microphone was found. Connect your microphone and click Refresh microphones.",
                OverconstrainedError: "The selected microphone is unavailable. Choose another microphone or reconnect it, then refresh the list.",
                NotReadableError: "The microphone could not be opened. Check its connection and whether another app is using it."
            };
            showStatus(messages[error.name] || "Could not start recording. Check your microphone and browser permission, then try again.", true);
        } finally {
            if (request === permissionRequest) {
                waitingPermission = false;
                busy = false;
                if (recorder) setStage("record");
                else updateControls();
            }
        }
    });

    function cancelPermission() {
        permissionRequest++;
        waitingPermission = false;
        busy = false;
        updateControls();
        showStatus(clip ? "Microphone request cancelled. Your previous recording is still available." : "Microphone request cancelled.");
    }

    function stopRecording() {
        if (!recorder || recorder.state === "inactive" || busy) return;
        busy = true;
        updateControls();
        holdClock();
        stopMeter();
        showStatus("Finishing recording…");
        recorder.stop();
    }

    cancel.addEventListener("click", cancelPermission);
    stop.addEventListener("click", stopRecording);

    async function withVoiceDeadline(work) {
        const controller = new AbortController();
        let timer;
        const timeout = new Promise((_, reject) => {
            timer = setTimeout(() => {
                controller.abort();
                reject(new Error("Saving took too long. Your recording is still here. Try Save Voice again."));
            }, 20000);
        });
        try {
            return await Promise.race([timeout, work(controller.signal)]);
        } finally {
            clearTimeout(timer);
        }
    }

    async function prepareWav(take) {
        if (take.wav) return take.wav;
        if (take.preparing) return take.preparing;
        let context;
        take.preparing = withVoiceDeadline(async signal => {
            context = new AudioContext(take.sampleRate ? {sampleRate: take.sampleRate} : {});
            const audio = await context.decodeAudioData(await take.blob.arrayBuffer());
            signal.throwIfAborted();
            take.wav = encodeVoiceWav(audio);
            return take.wav;
        });
        try {
            return await take.preparing;
        } finally {
            take.preparing = null;
            if (context) context.close().catch(() => {});
        }
    }

    async function uploadClip(take) {
        if (take.uploaded) return take.uploaded;
        return withVoiceDeadline(async signal => {
            const wav = await prepareWav(take);
            signal.throwIfAborted();
            const form = new FormData();
            form.append("files", wav, "voice.wav");
            const base = window.gradio_config.root.replace(/\/$/, "");
            const response = await fetch(`${base}/gradio_api/upload`, {
                method: "POST", body: form, signal
            });
            if (!response.ok) throw new Error("Upload failed. Your recording is still here. Try Save Voice again.");
            const paths = await response.json();
            if (!Array.isArray(paths) || typeof paths[0] !== "string") {
                throw new Error("The app could not accept the recording. Try Save Voice again.");
            }
            signal.throwIfAborted();
            take.uploaded = {
                path: paths[0], orig_name: "voice.wav", mime_type: "audio/wav",
                meta: {_type: "gradio.FileData"}
            };
            return take.uploaded;
        });
    }

    // Gradio marks its event pending after async preprocessing, so guard clicks here too.
    save.addEventListener("click", event => {
        if (busy || recorder || savePending || !clip) {
            event.preventDefault();
            event.stopImmediatePropagation();
            return;
        }
        savePending = true;
        saveResult.hidden = true;
    }, true);

    window.vibevoiceRecorder = {
        async prepareSave(inputs) {
            if (busy || recorder) return [null, ...inputs.slice(1)];
            if (!clip) {
                showStatus("Record your voice before saving.", true);
                return [null, ...inputs.slice(1)];
            }
            busy = true;
            updateControls();
            showStatus("Saving your recording…");
            try {
                const uploaded = await uploadClip(clip);
                return [uploaded, ...inputs.slice(1)];
            } catch (error) {
                showStatus(error.name === "AbortError"
                    ? "Saving took too long. Your recording is still here. Try Save Voice again."
                    : error.message, true);
                return [null, ...inputs.slice(1)];
            } finally {
                busy = false;
                updateControls();
            }
        },
        finishSave(saved, failed = false, file = null) {
            savePending = false;
            saveResult.hidden = failed;
            root.toggleAttribute("data-saved", saved === true);
            if (failed) showStatus("Could not save. Your recording is still available. Try Save Voice again.", true);
            else if (status.textContent === "Saving your recording…") showStatus("");
            if (saved === true) {
                savedDownload.hidden = !file?.url;
                if (file?.url) {
                    savedDownload.href = file.url;
                    savedDownload.download = file.orig_name || "voice.wav";
                } else {
                    savedDownload.removeAttribute("href");
                }
                clip = null;
                preview.pause();
                preview.removeAttribute("src");
                preview.load();
                if (previewUrl) URL.revokeObjectURL(previewUrl);
                previewUrl = null;
                showStatus("");
                if (root.open) {
                    if (root.dataset.stage !== "save") setStage("save");
                    done.focus();
                }
            }
            updateControls();
            return [];
        }
    };
    window.addEventListener("beforeunload", event => {
        if (clip || recorder) {
            event.preventDefault();
            event.returnValue = "";
        }
    });
    window.addEventListener("pagehide", () => {
        disposed = true;
        permissionRequest++;
        waitingPermission = false;
        busy = false;
        savePending = false;
        clearInterval(clock);
        holdClock();
        stopMeter();
        const active = recorder;
        if (active) {
            if (active.state !== "inactive") active.stop();
            active.stream.getTracks().forEach(track => track.stop());
        }
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    });
    window.addEventListener("pageshow", event => {
        if (!event.persisted) return;
        disposed = false;
        if (clip) {
            previewUrl = URL.createObjectURL(clip.blob);
            preview.src = previewUrl;
        }
        updateControls();
    });
    refresh.addEventListener("click", refreshMicrophones);
    navigator.mediaDevices?.addEventListener("devicechange", refreshMicrophones);
    refreshMicrophones();
    setStage("microphone");
}
