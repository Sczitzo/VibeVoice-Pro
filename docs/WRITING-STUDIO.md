# Writing Studio

Writing Studio refines this fork's existing composer into an editorial workspace. It changes the application interface, not the VibeVoice model or its training.

## What's included

- **A calmer writing space.** Warm paper styling, a Script label integrated into the card border, and an editable session title in the inner divider.
- **An editor that fits the window.** Text grows the writing area until it reaches the available space, then scrolls inside the editor. The player stays visible. Very short windows use a bounded middle-region scroller for expanded controls instead of scrolling the whole page.
- **Compact, clear voice assignments.** A numbered badge pairs each speaker with its selector; the accessible name still identifies the speaker. Narrow windows tuck assignments and advanced settings behind **Voices** without resetting selections. The dropdown arrow opens the picker.
- **Saved voices and built-in presets are separate.** Saved voices appear first; the protected **Built-in presets** group expands on demand. Removing a saved voice updates the picker without a reload and clears any selections using that voice. **Undo** restores the voice to the list. Other open tabs pick up changes when the picker is reopened.
- **Small tools, unchanged actions.** Dice cycles examples, the curved arrow undoes script changes, and the document-check icon validates the script. Each has an accessible name and tooltip. Numbered speaker buttons still insert the exact `Speaker N:` format.
- **Separate playback and generation information.** Preview elapsed time is distinct from estimated generation progress and time remaining. Remaining time is rounded for readability; the estimation algorithm is unchanged. **Stop generation** is separate from playback Pause.
- **Preserved playback behavior.** Pausing does not stop synthesis or permit a second concurrent generation. Finished audio retains the preview playhead and explicit pause intent, including buffering. A valid new generation clears the preceding live stream while retaining the completed take until replacement is ready.
- **Existing workflows stay available.** Takes, sections, Create Voice, and advanced settings remain in the app. Session titles stay in browser session storage and do not rename saved takes. System fonts and a bundled SVG mark require no external asset service.

## Verification

Run from the repository root in the project Python environment:

```sh
python -m unittest discover -s tests -v
node --test tests/*_js.cjs
```

Browser checks covered effective desktop widths from 320 to 1366 CSS pixels, short windows, editor growth and shrinkage, internal scrolling, keyboard example/Undo/validation actions, voice selection, and disclosure focus behavior. In the Codex browser, deletion and Undo were checked across two tabs using disposable test voices, including reopening a previously filtered picker. The default-collapsed **Built-in presets** group also opened on the live preview. No real saved voices were removed for testing.

Audio evidence is separate from layout evidence: earlier real local synthetic generation checks exercised playback and resizing, and a later short synthetic sample completed successfully. Active transport screenshots used a lightweight generation fixture, not model output. The latest icon, autosizing, and voice-row checks did not rerun synthesis. Estimated percentages and completion times are estimates, not promises or benchmark results.

These are local tests, not a claim that GitHub CI passed. They do not certify every operating-system scaling setting, embedded browser, microphone, or voice quality. The README screenshots show the idle studio with fictional example text and bundled voice labels. No private recordings, generated audio, user scripts, or local testing logs are included.

## Design and behavior contracts

Keep the textarea authoritative and its decorative text layer aligned while resizing and scrolling. Escape script content before rendering highlights. Preserve visible focus and the forced-colors text fallback. Do not clone voice selectors or reset their state when opening the disclosure. Pause controls playback only, not synthesis; a second generation must remain unavailable while one is running. Keep the Stop control rendered throughout generation because playback state detection depends on it.

This change builds on the existing composer work in `codex/vibevoice-pro`; the upstream model and community attribution remain in the main README.
