# Writing Studio

Writing Studio refines this fork's existing composer into an editorial workspace. It changes the application interface, not the VibeVoice model or its training.

## What changes

- Warm paper canvas, a single editable session title, compact voice assignments, and a restrained playback dock.
- At constrained window widths, **Voices** opens the existing speaker count, voice selectors, and advanced settings. Wide windows keep assignments inline.
- Small numbered insertion buttons retain the exact `Speaker N:` script format. Example cycling, Undo, validation, Takes, sections, and Create Voice remain available.
- Session titles stay in browser session storage; they do not rename saved takes.
- Playback preserves explicit pause intent through buffering and finished-audio handoff. Starting a valid new generation clears the preceding live stream while retaining the completed take until replacement is ready.
- System fonts and a bundled SVG mark require no external font or asset service.

## Verification

Run from the repository root in the project Python environment:

```sh
python -m unittest discover -s tests -v
node --test tests/*_js.cjs
```

The release candidate passed 59 Python and 57 JavaScript tests. Manual browser checks covered effective desktop widths from 320 to 1366 CSS pixels, including 695 by 763, keyboard voice selection, disclosure Escape/focus behavior, and resizing without losing script state. A real local generated sample retained its paused playhead during a wide-to-narrow resize.

These checks do not certify every operating-system scaling setting, embedded browser, microphone, or voice quality. Narrow windows use normal vertical scrolling instead of compressing the editor to zero height. No private recordings or generated audio are included.

## Design and behavior contracts

Keep the textarea authoritative and its decorative text layer aligned while resizing and scrolling. Escape script content before rendering highlights. Preserve visible focus and the forced-colors text fallback. Do not clone voice selectors or reset their state when opening the disclosure. Pause controls playback only, not synthesis; a second generation must remain unavailable while one is running.

This change builds on the existing composer work in `codex/vibevoice-pro`; the upstream model and community attribution remain in the main README.
