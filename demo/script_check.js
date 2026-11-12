function initScriptCheck() {
    const editor = document.querySelector("#composer-editor textarea");
    const speakers = document.getElementById("speaker-count");
    const feedback = document.getElementById("script-check-feedback");
    const actions = document.getElementById("composer-actions");
    const check = document.getElementById("check-script");
    const keep = document.getElementById("keep-speaker");
    if (!editor || !speakers || !feedback || !actions || !check || !keep) return;

    const read = () => ({ text: editor.value, count: speakers.querySelector("input:checked")?.value });
    const unchanged = ({ text, count }) => text === editor.value && count === read().count;
    const getResult = () => feedback.querySelector("[data-snapshot]");
    let explicitCheck = null;
    let revision = 0;

    async function currentResult() {
        const result = getResult();
        if (!result || !globalThis.crypto?.subtle) return null;
        const input = read();
        const data = new TextEncoder().encode(input.count + "\n" + input.text);
        const digest = await crypto.subtle.digest("SHA-256", data);
        const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
        return getResult() === result && unchanged(input) && hash === result.dataset.snapshot ? result : null;
    }

    function lineRange(line) {
        const separators = Array.from(editor.value.matchAll(/\r\n|[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]/g));
        const previous = separators[line - 2];
        const start = previous ? previous.index + previous[0].length : 0;
        return [start, separators[line - 1]?.index ?? editor.value.length];
    }

    async function refresh() {
        const requested = ++revision;
        const result = await currentResult();
        if (requested !== revision) return;
        const visible = getResult();
        if (visible) visible.hidden = visible !== result;
        const label = result?.dataset.label;
        keep.hidden = !label;
        if (label) keep.textContent = "Keep with Speaker " + label;
        const generate = document.getElementById("generate-audio");
        if (generate) generate.disabled = !!result?.classList.contains("script-check-issue");

        if (result && explicitCheck && unchanged(explicitCheck)) {
            explicitCheck = null;
            if (result.dataset.line) {
                editor.focus();
                editor.setSelectionRange(...lineRange(Number(result.dataset.line)));
            }
        }
    }

    check.addEventListener("click", () => {
        explicitCheck = read();
        void refresh();
    }, { capture: true });
    function onEdit() {
        explicitCheck = null;
        const result = getResult();
        if (result) result.hidden = true;
        keep.hidden = true;
        const generate = document.getElementById("generate-audio");
        if (generate) generate.disabled = false;
        void refresh();
    }
    editor.addEventListener("input", onEdit);
    speakers.addEventListener("change", onEdit);

    keep.addEventListener("click", async () => {
        const shown = getResult();
        const input = read();
        if (!shown || shown.hidden || !shown.dataset.label) return;
        const result = await currentResult();
        // The digest is asynchronous. Never edit the textarea using an older response or input.
        if (result !== shown || !unchanged(input)) return;
        const [start, end] = lineRange(Number(result.dataset.line));
        const indentation = /^[ \t]*/.exec(editor.value.slice(start, end))[0].length;
        const offset = start + indentation;
        editor.focus();
        editor.setRangeText("Speaker " + result.dataset.label + ": ", offset, offset, "end");
        editor.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const observer = new MutationObserver(() => { void refresh(); });
    observer.observe(feedback, { childList: true, subtree: true });
    observer.observe(actions, { childList: true, subtree: true });
}
