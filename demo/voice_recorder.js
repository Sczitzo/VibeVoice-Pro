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
    let mode = "record";
    let imported = null;
    let importVersion = 0;
    let importLoading = false;
    let rangePlayer = null;
    const modes = [...root.querySelectorAll("[data-voice-mode]")];
    const sourceModes = root.querySelector(".voice-modes");
    const inputFile = root.querySelector("#voice-import-file");
    const range = root.querySelector("[data-import-range]");
    const start = root.querySelector("[data-import-start]");
    const end = root.querySelector("[data-import-end]");
    const playRange = root.querySelector("[data-import-preview]");
    const nameRange = root.querySelector("[data-import-name]");
    const draftNames = {record: "", import: ""};

    const bounds = () => {
        const from = start?.valueAsNumber, to = end?.valueAsNumber;
        return imported && !importLoading && Number.isFinite(from) && Number.isFinite(to) &&
            from >= 0 && to > from && to <= imported.duration &&
            Math.round(to * imported.rate) > Math.round(from * imported.rate)
            ? [Math.round(from * imported.rate) / imported.rate, Math.round(to * imported.rate) / imported.rate] : null;
    };
    function stopRange() {
        rangePlayer?.source.stop();
        if (rangePlayer) rangePlayer.context.close().catch(() => {});
        rangePlayer = null;
        if (playRange) playRange.textContent = "Preview range";
    }
    function drawRange() {
        if (!imported) return;
        const canvas = range.querySelector("canvas");
        const draw = canvas.getContext("2d");
        const {width, height} = canvas;
        const {peaks, duration} = imported;
        const selected = bounds();
        range.querySelector("[data-import-duration]").textContent = selected
            ? `${(selected[1] - selected[0]).toFixed(3)} seconds selected` : "";
        draw.clearRect(0, 0, width, height);
        if (selected) {
            draw.fillStyle = "#d2e9de";
            draw.fillRect(selected[0] / duration * width, 0,
                (selected[1] - selected[0]) / duration * width, height);
        }
        draw.fillStyle = "#176458";
        peaks.forEach((peak, x) => draw.fillRect(x, height / 2 - peak * height / 2, 1, Math.max(1, peak * height)));
        updateControls();
    }
    inputFile?.addEventListener("change", async () => {
        const file = inputFile.files[0];
        if (!file) return;
        const version = ++importVersion;
        stopRange();
        importLoading = true;
        updateControls();
        let context;
        try {
            if (file.size > 32 * 1024 * 1024) throw new Error("Choose a WAV no larger than 32 MiB.");
            const bytes = await file.arrayBuffer();
            const decoder = new TextDecoder();
            const header = decoder.decode(bytes.slice(0, 12));
            if (!header.startsWith("RIFF") || header.slice(8) !== "WAVE") throw new Error("Choose a valid WAV file.");
            const view = new DataView(bytes);
            let rate, frameBytes, format, dataBytes;
            for (let offset = 12; offset + 8 <= bytes.byteLength;) {
                const type = decoder.decode(bytes.slice(offset, offset + 4));
                const size = view.getUint32(offset + 4, true);
                if (size > bytes.byteLength - offset - 8) throw new Error("The WAV file is incomplete.");
                if (type === "fmt " && size >= 16) {
                    format = view.getUint16(offset + 8, true);
                    rate = view.getUint32(offset + 12, true);
                    frameBytes = view.getUint16(offset + 20, true);
                }
                if (type === "data") dataBytes = size;
                offset += 8 + size + (size % 2);
                if (rate && dataBytes !== undefined) break;
            }
            if (!rate || !frameBytes || !dataBytes) throw new Error("Choose a WAV file with audio.");
            const originalDuration = [1, 3, 65534].includes(format) ? Math.floor(dataBytes / frameBytes) / rate : null;
            if (originalDuration > 60) throw new Error("Choose a WAV up to 60 seconds long.");
            try { context = new AudioContext({sampleRate: rate}); }
            catch { context = new AudioContext(); }
            const audio = await withVoiceDeadline(() => context.decodeAudioData(bytes),
                "Opening the WAV took too long. Choose the file again.");
            if (version !== importVersion || disposed) return;
            const duration = originalDuration ?? audio.duration;
            if (audio.numberOfChannels > 2 || duration > 60 || !audio.length) {
                throw new Error("Choose a mono or stereo WAV up to 60 seconds long.");
            }
            const width = range.querySelector("canvas").width;
            const peaks = Array(width).fill(0);
            for (let channel = 0; channel < audio.numberOfChannels; channel++) {
                const data = audio.getChannelData(channel);
                for (let x = 0; x < width; x++) {
                    const stopAt = Math.floor((x + 1) * data.length / width);
                    for (let i = Math.floor(x * data.length / width); i < stopAt; i++) {
                        peaks[x] = Math.max(peaks[x], Math.min(1, Math.abs(data[i])));
                    }
                }
            }
            imported = {wav: file, audio, peaks, duration, rate};
            range.querySelector("[data-import-filename]").textContent = file.name;
            start.value = "0";
            end.value = String(duration);
            range.hidden = false;
            if (mode === "import") showStatus("");
        } catch (error) {
            if (version === importVersion) {
                inputFile.value = "";
                if (mode === "import") showStatus((error.message || "The WAV could not be opened.")
                    + (imported ? " The previous selection is still available." : ""), true);
            }
        } finally {
            if (version === importVersion) {
                importLoading = false;
                drawRange();
                updateControls();
            }
            if (context) context.close().catch(() => {});
        }
    });
    for (const field of [start, end]) field?.addEventListener("input", () => {
        stopRange();
        drawRange();
        showStatus(bounds() ? "" : "Choose a start and end inside the WAV.", !bounds());
    });
    playRange?.addEventListener("click", () => {
        if (rangePlayer) return stopRange();
        const selected = bounds();
        if (!selected) return;
        const context = new AudioContext();
        const source = context.createBufferSource();
        source.buffer = imported.audio;
        source.connect(context.destination);
        rangePlayer = {source, context};
        source.onended = () => { if (rangePlayer?.source === source) stopRange(); };
        source.start(0, selected[0], selected[1] - selected[0]);
        playRange.textContent = "Stop preview";
    });
    nameRange?.addEventListener("click", () => {
        if (!bounds()) return;
        stopRange();
        root.removeAttribute("data-saved");
        saveResult.hidden = true;
        showStatus("");
        setStage("save");
    });
    for (const button of modes) button.addEventListener("click", () => {
        if (mode === button.dataset.voiceMode || busy || savePending || recorder || waitingPermission) return;
        draftNames[mode] = voiceName.value;
        mode = button.dataset.voiceMode;
        root.dataset.voiceMode = mode;
        voiceName.value = draftNames[mode];
        voiceName.dispatchEvent(new Event("input", {bubbles: true}));
        for (const option of modes) option.setAttribute("aria-pressed", String(option === button));
        root.removeAttribute("data-saved");
        preview.pause();
        stopRange();
        showStatus("");
        setStage(mode === "import" ? "import" : clip ? "listen" : "microphone");
    });

    function setStage(stage) {
        root.dataset.stage = stage;
        for (const pane of panes) pane.hidden = pane.dataset.pane !== stage;
        for (const step of steps) {
            if (step.dataset.step === stage) step.setAttribute("aria-current", "step");
            else step.removeAttribute("aria-current");
        }
        updateControls();
        if (root.open) {
            const control = {microphone, record: readingTitle, listen: preview, save: voiceName, import: inputFile}[stage];
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
        // Source choice belongs at the initial step, not over an active or retained take.
        if (sourceModes) sourceModes.hidden = waitingPermission ||
            ["record", "listen", "save"].includes(root.dataset.stage);
        record.disabled = locked || recorder !== null;
        stop.disabled = busy || recorder === null;
        pause.disabled = stop.disabled;
        pause.textContent = recorder?.state === "paused" ? "Resume" : "Pause";
        cancel.hidden = !waitingPermission;
        returnToTake.hidden = !clip || waitingPermission;
        returnToTake.disabled = locked;
        microphone.disabled = record.disabled;
        refresh.disabled = record.disabled;
        const canSave = mode === "import" ? Boolean(bounds()) : Boolean(clip);
        save.disabled = locked || recorder !== null || !canSave;
        back.disabled = locked || !canSave;
        back.textContent = mode === "import" ? "Back to range" : "Back to Listen";
        for (const option of modes) option.disabled = locked || Boolean(recorder) || waitingPermission;
        if (playRange) playRange.disabled = !canSave;
        if (nameRange) nameRange.disabled = !canSave;
        if (inputFile) inputFile.disabled = locked;
    }

    openDialog.addEventListener("click", () => {
        if (root.open) return;
        if (!recorder && !waitingPermission) {
            if (savePending) setStage("save");
            else setStage(mode === "import" ? "import" : clip ? "listen" : "microphone");
        }
        root.showModal();
        if (root.dataset.stage === "microphone") microphone.focus();
        if (root.dataset.stage === "listen") preview.focus();
        if (root.dataset.stage === "save") voiceName.focus();
        if (root.dataset.stage === "import") inputFile.focus();
        refreshMicrophones();
    });

    function dismissRecorder() {
        if (waitingPermission) cancelPermission();
        if (recorder && recorder.state !== "inactive" && !busy) stopRecording();
        preview.pause();
        stopRange();
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
    back.addEventListener("click", () => setStage(mode === "import" ? "import" : "listen"));
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

    async function withVoiceDeadline(work, message = "Saving took too long. Your recording is still here. Try Save Voice again.") {
        const controller = new AbortController();
        let timer;
        const timeout = new Promise((_, reject) => {
            timer = setTimeout(() => {
                controller.abort();
                reject(new Error(message));
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
        if (busy || recorder || savePending || !(mode === "import" ? bounds() : clip)) {
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
            const selection = mode === "import" ? bounds() : null;
            const take = mode === "import" ? imported : clip;
            if (!take || (mode === "import" && !selection)) {
                showStatus("Record your voice before saving.", true);
                return [null, ...inputs.slice(1)];
            }
            busy = true;
            updateControls();
            showStatus("Saving your recording…");
            try {
                const uploaded = await uploadClip(take);
                return selection ? [uploaded, inputs[1], ...selection] : [uploaded, ...inputs.slice(1)];
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
                if (mode === "import") {
                    imported = null;
                    inputFile.value = "";
                    range.hidden = true;
                } else {
                    clip = null;
                    preview.pause();
                    preview.removeAttribute("src");
                    preview.load();
                    if (previewUrl) URL.revokeObjectURL(previewUrl);
                    previewUrl = null;
                }
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
        if (clip || imported || recorder) {
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
        stopRange();
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
