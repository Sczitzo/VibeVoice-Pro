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
    const number = new Intl.NumberFormat();
    let editorUsed = false;

    function refresh() {
        const text = editor.value.replace(/^\s*Speaker[ \t]+\p{Nd}+[ \t]*:/gimu, "").trim();
        const total = text ? text.split(/\s+/u).length : 0;
        words.textContent = number.format(total) + (total === 1 ? " word" : " words");
        const speakers = Number(count.querySelector("input:checked")?.value);
        for (const chip of chips) chip.hidden = Number(chip.dataset.speaker) > speakers;
    }

    editor.addEventListener("focus", () => { editorUsed = true; });
    editor.addEventListener("input", refresh);
    count.addEventListener("change", refresh);
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
    for (const input of settingsFields.querySelectorAll("#guidance-strength input")) {
        input.setAttribute("aria-label", "Guidance Strength");
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
        refreshExample() {
            editorUsed = false;
            refresh();
        },
    };
    refresh();
}
