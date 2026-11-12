function initStudioAudio() {
    const app = document.getElementById("guided-composer");
    if (!app || app.dataset.studioAudio) return;
    app.dataset.studioAudio = "ready";
    const dialog = document.getElementById("studio-library");
    const opener = document.getElementById("open-studio-library");
    const fields = document.getElementById("studio-library-fields");
    if (!dialog || !opener) return;
    if (!fields) { opener.hidden = true; return; }
    document.getElementById("studio-library-fields-host").append(fields);
    if (fields.classList.contains("hidden")) { opener.hidden = true; return; }
    const stop = document.getElementById("stop-generation");
    const stopHere = dialog.querySelector("[data-stop-studio]");
    const feedback = document.getElementById("generation-feedback");
    const status = dialog.querySelector("[data-studio-status]");
    const sync = () => {
        stopHere.disabled = !stop || stop.disabled || stop.getClientRects().length === 0;
        status.textContent = feedback?.getClientRects().length ? feedback.textContent : "";
    };
    stopHere.addEventListener("click", () => { if (!stopHere.disabled) stop.click(); });
    const observer = new MutationObserver(sync);
    observer.observe(app, {subtree: true, childList: true, attributes: true, characterData: true});
    window.addEventListener("pagehide", () => observer.disconnect(), {once: true});
    sync();
    opener.addEventListener("click", () => dialog.showModal());
    dialog.querySelector("[data-close-studio]").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
        for (const audio of dialog.querySelectorAll("audio")) audio.pause();
        opener.focus();
    });
    document.addEventListener("click", async event => {
        if (event.target.closest("button[data-section-command]") && window.vibevoicePlayback?.isPending()) {
            event.preventDefault();
            event.stopImmediatePropagation();
            return;
        }
        if (event.target.closest("#generate-audio, button[data-section-command]")) {
            const live = document.querySelector("#live-generated-audio audio");
            if (live) live.muted = false;
        }
        const action = event.target.closest("button[data-section-command]");
        if (action) {
            const field = document.querySelector("#section-command textarea, #section-command input");
            const send = document.getElementById("section-action");
            if (!field || !send) return;
            field.value = action.dataset.sectionCommand;
            field.dispatchEvent(new Event("input", {bubbles: true}));
            action.disabled = true;
            if (action.dataset.sectionCommand.endsWith(":export")) window.vibevoicePlayback?.acceptExport();
            else window.vibevoicePlayback?.start();
            requestAnimationFrame(() => send.click());
            return;
        }
        const preview = event.target.closest("button[data-section-preview]");
        if (preview) {
            const audio = preview.closest(".studio-audio-row").querySelector("audio");
            const base = window.gradio_config.root.replace(/\/$/, "");
            for (const other of document.querySelectorAll(".studio-audio-row audio")) if (other !== audio) other.pause();
            audio.src = `${base}/gradio_api/file=${encodeURIComponent(preview.dataset.sectionPreview)}`;
            audio.hidden = false;
            try {
                await audio.play();
            } catch {
                const status = document.querySelector("#generation-sections [role=status]");
                if (status) status.textContent = "Could not play this section. Export checks whether the section needs a retry.";
            }
            return;
        }
        const button = event.target.closest("button[data-take-id]");
        if (!button) return;
        const row = button.closest(".studio-audio-row");
        const status = document.getElementById("take-action-status");
        const base = window.gradio_config.root.replace(/\/$/, "");
        button.disabled = true;
        try {
            const response = await fetch(`${base}/gradio_api/run/open_saved_take`, {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({data: [button.dataset.takeId]})
            });
            const result = await response.json();
            const file = result.data?.[0];
            if (!response.ok || !file?.url) throw new Error("This saved take is no longer available.");
            status.textContent = "";
            if (button.dataset.takeAction === "download") {
                const link = document.createElement("a");
                link.href = file.url;
                link.download = file.orig_name || "take.wav";
                link.click();
            } else {
                const audio = row.querySelector("audio");
                for (const other of document.querySelectorAll(".studio-audio-row audio")) if (other !== audio) other.pause();
                audio.src = file.url;
                audio.hidden = false;
                await audio.play();
            }
        } catch (error) {
            status.textContent = error.message || "Could not open the saved take.";
        } finally {
            button.disabled = false;
        }
    });
}
