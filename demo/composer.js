// Local editor history includes programmatic replacements as well as typing.
function createScriptHistory(initial, apply, changed) {
    let present = initial;
    const past = [], future = [];
    const notify = () => changed(past.length > 0);
    return {
        record(next) {
            if (next.text === present.text) { present = next; return; }
            past.push(present);
            present = next;
            future.length = 0;
            notify();
        },
        undo() {
            if (!past.length) return;
            future.push(present);
            present = past.pop();
            apply(present);
            notify();
        },
        redo() {
            if (!future.length) return;
            past.push(present);
            present = future.pop();
            apply(present);
            notify();
        },
    };
}

function initComposer() {
    const shell = document.getElementById("guided-composer");
    const editor = document.querySelector("#composer-editor textarea");
    const count = document.getElementById("speaker-count");
    const chips = [...document.querySelectorAll("#speaker-chips [data-speaker]")];
    const words = document.getElementById("word-count");
    const panel = document.getElementById("show-speakers");
    const settings = document.getElementById("composer-settings");
    const settingsButton = document.getElementById("open-advanced-settings");
    const settingsFields = document.getElementById("composer-settings-fields");
    // Match the border opening to the actual title, including its help control.
    for (const [panelId, labelSelector] of [
        ["composer-script", "#composer-script-label"],
        ["composer-speakers", ".speaker-heading-label"],
    ]) {
        const box = document.getElementById(panelId);
        const label = box?.querySelector(labelSelector);
        if (!box || !label) continue;
        const updateGap = () => box.style.setProperty("--heading-gap", `${label.getBoundingClientRect().width}px`);
        updateGap();
        if (typeof ResizeObserver !== "undefined") new ResizeObserver(updateGap).observe(label);
    }
    const number = new Intl.NumberFormat();
    let editorUsed = false;
    const undo = document.getElementById("undo-script");
    let restoring = false;
    const snapshot = () => ({text: editor.value, start: editor.selectionStart,
        end: editor.selectionEnd, speakers: count.querySelector("input:checked")?.value});
    function restore(state) {
        restoring = true;
        try {
            const choice = [...count.querySelectorAll("input")].find(input => input.value === String(state.speakers));
            if (choice && !choice.checked) choice.click();
            editor.value = state.text;
            editor.dispatchEvent(new Event("input", {bubbles: true}));
            editor.focus();
            editor.setSelectionRange(state.start ?? 0, state.end ?? 0);
            editorUsed = true;
        } finally { restoring = false; }
        refreshAfterRender();
    }
    const history = createScriptHistory(snapshot(), restore, available => {
        undo.hidden = !available;
        undo.disabled = !available;
    });
    editor.addEventListener("input", event => {
        if (!restoring && !event.isComposing) history.record(snapshot());
    });
    editor.addEventListener("compositionend", () => { if (!restoring) history.record(snapshot()); });
    editor.addEventListener("keydown", event => {
        if (!(event.ctrlKey || event.metaKey) || event.altKey || event.isComposing) return;
        const key = event.key.toLowerCase();
        if (key === "z" || (key === "y" && event.ctrlKey)) {
            event.preventDefault();
            if (event.shiftKey || key === "y") history.redo();
            else history.undo();
        }
    });
    editor.addEventListener("beforeinput", event => {
        if (event.inputType === "historyUndo" || event.inputType === "historyRedo") {
            event.preventDefault();
            if (event.inputType === "historyUndo") history.undo();
            else history.redo();
        } else if (!restoring && !event.isComposing) {
            // Remember the caret before typing/paste, including a recent pointer move.
            history.record(snapshot());
        }
    });
    undo.addEventListener("click", () => history.undo());

    function refresh() {
        const text = editor.value.replace(/^\s*Speaker[ \t]+\p{Nd}+[ \t]*:/gimu, "").trim();
        const total = text ? text.split(/\s+/u).length : 0;
        words.textContent = number.format(total) + (total === 1 ? " word" : " words");
        const speakers = Number(count.querySelector("input:checked")?.value);
        for (const chip of chips) chip.hidden = Number(chip.dataset.speaker) > speakers;
    }

    editor.addEventListener("focus", () => { editorUsed = true; });
    editor.addEventListener("input", refresh);
    // Gradio may flush radio state after its change handler. Re-read the current
    // selection after rendering so Undo/Redo and rapid changes cannot leave stale chips.
    function refreshAfterRender() {
        refresh();
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(refresh);
    }
    count.addEventListener("change", refreshAfterRender);
    for (const chip of chips) {
        chip.addEventListener("click", () => {
            const { value } = editor;
            const caret = editorUsed ? editor.selectionStart : value.length;
            const line = value.lastIndexOf("\n", caret - 1) + 1;
            const match = /^[ \t]*Speaker[ \t]+\p{Nd}+[ \t]*:[ \t]*/iu.exec(value.slice(line));
            const end = line + (match?.[0].length ?? 0);
            const zeroBased = /^[ \t]*Speaker[ \t]+[0０]+[ \t]*:/imu.test(value);
            const label = "Speaker " + (Number(chip.dataset.speaker) - Number(zeroBased)) + ": ";
            if (editorUsed && match && editor.selectionEnd <= end) {
                editor.setRangeText(label, line, end, "end");
            } else {
                editor.setRangeText((caret > line ? "\n\n" : "") + label, caret, caret, "end");
            }
            editor.focus();
            editor.dispatchEvent(new Event("input", { bubbles: true }));
        });
    }

    function showSpeakers(show) {
        shell.dataset.panel = show ? "speakers" : "script";
        panel.setAttribute("aria-expanded", String(show));
        panel.textContent = show ? "Back to Script" : "Speakers";
        if (show) count.querySelector("input:checked")?.focus();
        else editor.focus();
    }

    panel.addEventListener("click", () => showSpeakers(shell.dataset.panel !== "speakers"));
    document.getElementById("back-to-script").addEventListener("click", () => showSpeakers(false));
    const refreshVoices = document.getElementById("refresh-voices");
    if (refreshVoices && typeof BroadcastChannel === "function") {
        const voiceUpdates = new BroadcastChannel("vibevoice-saved-voices");
        const assignments = document.getElementById("voice-assignments");
        const pickerSelector = 'input[aria-controls="dropdown-options"]';
        const catalog = document.getElementById("voice-menu-catalog");
        let refreshPending = false;
        let closedPicker;
        let priorCatalogUpdate;
        let reopen = false;

        function refreshClosedMenus() {
            if (!refreshPending || assignments.querySelector(`${pickerSelector}[aria-expanded="true"]`)) return;
            refreshPending = false;
            const focused = document.activeElement;
            closedPicker = assignments.contains(focused) && focused.matches(pickerSelector) ? focused : null;
            reopen = false;
            closedPicker?.blur();
            priorCatalogUpdate = catalog.querySelector("[data-saved-voice-catalog]")?.dataset.update;
            refreshVoices.click();
        }
        voiceUpdates.addEventListener("message", () => {
            refreshPending = true;
            refreshClosedMenus();
        });
        new MutationObserver(refreshClosedMenus).observe(assignments, {
            subtree: true, childList: true, attributes: true, attributeFilter: ["aria-expanded"],
        });
        assignments.addEventListener("keydown", event => {
            if (event.target !== closedPicker) return;
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                reopen = true;
                event.preventDefault();
                event.stopImmediatePropagation();
            }
            if (event.key === "Escape") reopen = false;
        }, true);
        new MutationObserver(() => {
            const picker = closedPicker;
            if (!picker || catalog.querySelector("[data-saved-voice-catalog]")?.dataset.update === priorCatalogUpdate) return;
            // Gradio filters updated choices by the selected label while focused.
            // Wait for the updated choices before using Gradio's focus handler to restore its full list.
            setTimeout(() => {
                if (closedPicker !== picker) return;
                closedPicker = null;
                if (document.activeElement !== picker && document.activeElement !== document.body) return;
                picker.focus();
                picker.dispatchEvent(new FocusEvent("focus"));
                if (!reopen) {
                    picker.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape", bubbles: true}));
                }
            });
        }).observe(catalog, {subtree: true, childList: true, attributes: true, attributeFilter: ["data-saved-voice-catalog", "data-update"]});
        document.getElementById("voice-recorder").addEventListener("vibevoice-voice-saved", () => {
            voiceUpdates.postMessage("saved");
        });
        assignments.addEventListener("vibevoice-voices-changed", () => voiceUpdates.postMessage("changed"));
    }
    shell.querySelector(".composer-skip").addEventListener("click", event => {
        event.preventDefault();
        showSpeakers(false);
    });

    for (const help of document.querySelectorAll(".composer-help")) {
        const button = help.querySelector("button");
        const tip = help.querySelector('[role="tooltip"]');
        let pinned = false;
        const show = () => { tip.hidden = false; };
        const close = () => { tip.hidden = true; pinned = false; };
        help.addEventListener("pointerenter", event => {
            if (event.pointerType === "mouse") show();
        });
        help.addEventListener("pointerleave", () => {
            if (!pinned && !button.matches(":focus-visible")) close();
        });
        button.addEventListener("focus", () => {
            if (button.matches(":focus-visible")) show();
        });
        button.addEventListener("blur", close);
        button.addEventListener("click", () => {
            if (pinned) close();
            else {
                pinned = true;
                show();
            }
        });
        document.addEventListener("pointerdown", event => {
            if (!help.contains(event.target)) close();
        });
        document.addEventListener("keydown", event => {
            if (event.key === "Escape" && !tip.hidden) {
                event.preventDefault();
                event.stopPropagation();
                close();
            }
        }, true);
    }

    document.getElementById("settings-fields-host").append(settingsFields);
    const guidance = settingsFields.querySelector("#guidance-strength");
    guidance.querySelector(".head label")?.after(document.getElementById("guidance-help"));
    for (const input of guidance.querySelectorAll("input")) {
        input.setAttribute("aria-describedby", "guidance-tooltip");
    }
    settingsButton.addEventListener("click", () => settings.showModal());
    settings.querySelector("[data-close-settings]").addEventListener("click", () => settings.close());
    settings.addEventListener("close", () => {
        settings.querySelector(".composer-tooltip").hidden = true;
        settingsButton.focus();
    });
    editor.setAttribute("autocomplete", "off");
    editor.setAttribute("name", "script");
    window.vibevoiceComposer = {
        applyExample(example) {
            if (typeof example === "string") {
                try { example = JSON.parse(example); } catch { return; }
            }
            if (!example || typeof example.script !== "string") return;
            // Capture even edits made while the server was selecting an example.
            history.record(snapshot());
            restore({text: example.script, speakers: example.speakers, start: 0, end: 0});
            history.record(snapshot());
            editorUsed = false;
            refresh();
        },
        refreshExample() {
            editorUsed = false;
            refresh();
        },
    };
    refresh();
}
