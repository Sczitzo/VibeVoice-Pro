function initVoiceMenu() {
    const assignments = document.getElementById("voice-assignments");
    const catalog = document.getElementById("voice-menu-catalog");
    const feedback = document.getElementById("voice-menu-feedback");
    const status = feedback.querySelector('[role="status"]');
    const undo = feedback.querySelector("button");
    const announcement = document.getElementById("voice-menu-announcement");
    const inputs = 'input[aria-controls="dropdown-options"]';
    const options = '[data-testid="dropdown-option"]';
    const icons = {
        more: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>',
        trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"/></svg>',
    };
    let armed;
    let request;

    function paint() {
        const names = new Set(JSON.parse(catalog.querySelector('[data-saved-voice-names]')?.dataset.savedVoiceNames || '[]'));
        for (const input of assignments.querySelectorAll(inputs)) {
            input.setAttribute("aria-describedby", "voice-menu-help");
            input.placeholder = "Choose a voice";
        }
        for (const row of assignments.querySelectorAll(options)) {
            const name = row.getAttribute("aria-label");
            const item = row.closest(".speaker-item");
            const open = item.querySelector(inputs)?.getAttribute("aria-expanded") === "true";
            let button = row.querySelector(".voice-row-action");
            if (!open || !names.has(name)) {
                button?.remove();
                continue;
            }
            if (!button) {
                button = document.createElement("button");
                button.type = "button";
                button.className = "voice-row-action";
                // The listbox keeps keyboard focus. Shift+F10 exposes the same action.
                button.tabIndex = -1;
                button.setAttribute("aria-hidden", "true");
                row.append(button);
            }
            const action = armed?.item === item && armed.name === name ? "trash" : "more";
            if (button.dataset.action !== action || button.dataset.voice !== name) {
                button.dataset.action = action;
                button.dataset.voice = name;
                button.title = action === "trash" ? `Delete ${name}` : `Actions for ${name}`;
                button.innerHTML = icons[action];
            }
            button.disabled = Boolean(request);
        }
    }

    const observer = new MutationObserver(paint);
    observer.observe(assignments, {subtree: true, childList: true, attributes: true,
        attributeFilter: ["aria-label", "aria-expanded"]});
    observer.observe(catalog, {subtree: true, childList: true});
    paint();

    function arm(row) {
        const name = row.getAttribute("aria-label");
        armed = {item: row.closest(".speaker-item"), name};
        announcement.textContent = `Delete ${name}. Press Enter to remove or Escape to cancel.`;
        paint();
    }

    function disarm() {
        if (!armed) return;
        armed = undefined;
        announcement.textContent = "";
        paint();
    }

    function submit(command, input) {
        if (request) return;
        request = {input, reopen: input?.getAttribute("aria-expanded") === "true" || command === "undo"};
        input?.blur();
        disarm();
        undo.disabled = true;
        const field = document.querySelector("#voice-menu-command textarea, #voice-menu-command input");
        field.value = command;
        field.dispatchEvent(new Event("input", {bubbles: true}));
        requestAnimationFrame(() => document.getElementById("voice-menu-submit").click());
    }

    assignments.addEventListener("mousedown", event => {
        if (!event.target.closest(options) || (event.button === 0 && !event.target.closest(".voice-row-action"))) return;
        event.preventDefault();
        event.stopPropagation();
    }, true);
    assignments.addEventListener("contextmenu", event => {
        const row = event.target.closest(options);
        if (!row?.querySelector(".voice-row-action") || request) return;
        event.preventDefault();
        arm(row);
    });
    assignments.addEventListener("click", event => {
        const button = event.target.closest(".voice-row-action");
        if (!button || request) return;
        event.preventDefault();
        event.stopPropagation();
        if (button.dataset.action === "trash") {
            submit(`delete:${button.dataset.voice}`, button.closest(".speaker-item").querySelector(inputs));
        } else arm(button.closest(options));
    }, true);
    assignments.addEventListener("keydown", event => {
        const input = event.target;
        if (!input.matches(inputs) || input.getAttribute("aria-expanded") !== "true") return;
        const item = input.closest(".speaker-item");
        if (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10")) {
            const row = item.querySelector(`${options}.active`) || item.querySelector(`${options}[aria-selected="true"]`);
            if (!row?.querySelector(".voice-row-action") || request) return;
            arm(row);
        } else if (armed?.item === item && event.key === "Enter" && !request) {
            submit(`delete:${armed.name}`, input);
        } else if (armed?.item === item && event.key === "Escape") {
            disarm();
        } else {
            disarm();
            return;
        }
        event.preventDefault();
        event.stopImmediatePropagation();
    }, true);
    assignments.addEventListener("focusout", disarm);

    let lastInput;
    undo.addEventListener("click", () => submit("undo", lastInput));
    window.vibevoiceVoiceMenu = {
        finish(result) {
            if (!request) return;
            const {input, reopen} = request;
            request = undefined;
            if (input) lastInput = input;
            const hadUndo = !undo.hidden;
            const slots = (result?.slots || []).filter(slot =>
                assignments.querySelector(`.tone-${slot} ${inputs}`)?.getClientRects().length);
            const replacement = slots.length ? ` Choose a voice for ${slots.map(slot => `Speaker ${slot}`).join(" and ")}.` : "";
            feedback.hidden = false;
            feedback.classList.toggle("voice-menu-error", !result || Boolean(result.error));
            status.textContent = (result?.message || "Could not update saved voices. Try again.") + replacement;
            undo.hidden = !(result?.undo ?? hadUndo);
            undo.disabled = false;
            if (result?.changed) assignments.dispatchEvent(new Event("vibevoice-voices-changed"));
            paint();
            if (reopen && input?.isConnected && (document.activeElement === document.body || document.activeElement === undo)) {
                // Let Gradio apply the new choices before opening the full list again.
                requestAnimationFrame(() => {
                    if (document.activeElement === document.body || document.activeElement === undo) input.focus();
                });
            }
        },
    };
    const resultField = document.getElementById("voice-menu-result");
    new MutationObserver(() => {
        const payload = resultField.querySelector("[data-voice-result]")?.dataset.voiceResult;
        if (payload) window.vibevoiceVoiceMenu.finish(JSON.parse(payload));
    }).observe(resultField, {subtree: true, childList: true});
}
