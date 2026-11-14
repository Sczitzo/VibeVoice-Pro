function writingStudioMarkup(text) {
    const escape = value => value.replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    return String(text).split('\n').map(line => {
        const match = /^([ \t]*)(Speaker[ \t]+([0-9]+)[ \t]*:)(.*)$/i.exec(line);
        return match ? escape(match[1]) + '<mark class="speaker-highlight tone-' + ((Number(match[3]) || 1) % 5) + '">' + escape(match[2]) + '</mark>' + escape(match[4]) : escape(line);
    }).join('\n') + '\n';
}
function initWritingStudio() {
    const title = document.getElementById('manuscript-title');
    if (!title) return;
    // This labels the current browser draft, not a saved take or published project.
    try { title.value = sessionStorage.getItem('vibevoice-draft-title') || ''; } catch (_) {}
    const sync = () => {
        try { sessionStorage.setItem('vibevoice-draft-title', title.value); } catch (_) {}
    };
    title.addEventListener('input', sync);
    sync();
    for (const [id, name] of [['random-example', 'Random Example'], ['check-script', 'Check Script'], ['undo-script', 'Undo']]) {
        const control = document.getElementById(id);
        if (control?.setAttribute) { control.setAttribute('title', name); control.setAttribute('aria-label', name); }
    }
    initCastDisclosure();
    initVoiceInputPresentation();
    initAudioLegend();
    const editor = document.querySelector('#composer-editor textarea');
    if (!editor) return;
    const layer = document.createElement('pre');
    layer.className = 'manuscript-ink';
    layer.setAttribute('aria-hidden', 'true');
    editor.parentElement.classList.add('manuscript-edit-surface');
    editor.before(layer);
    const paint = () => {
        layer.style.width = editor.clientWidth + 'px';
        layer.innerHTML = writingStudioMarkup(editor.value);
        layer.scrollTop = editor.scrollTop; layer.scrollLeft = editor.scrollLeft;
    };
    editor.addEventListener('input', paint);
    editor.addEventListener('scroll', paint);
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(paint).observe(editor);
    paint();
    initWritingStudioAutosize(editor, paint);
}

function initCastDisclosure() {
    const cast = document.getElementById('composer-speakers');
    if (!cast || document.getElementById('cast-disclosure')) return;
    const button = document.createElement('button');
    button.id = 'cast-disclosure'; button.type = 'button';
    button.setAttribute('aria-controls', 'speaker-heading voice-assignments speaker-actions');
    cast.prepend(button);
    const media = window.matchMedia('(max-width: 900px)');
    let expanded = false;
    const update = () => {
        const count = cast.querySelector('#speaker-count input:checked')?.value || '2';
        button.textContent = 'Voices · ' + count + (expanded ? '  −' : '  +');
        button.setAttribute('aria-expanded', String(!media.matches || expanded));
        cast.classList.toggle('cast-expanded', expanded);
    };
    button.addEventListener('click', () => { expanded = !expanded; update(); });
    cast.addEventListener('change', update);
    cast.addEventListener('keydown', event => {
        if (event.key === 'Escape' && media.matches && expanded) {
            expanded = false; update(); button.focus(); event.preventDefault();
        }
    });
    media.addEventListener('change', () => {
        if (media.matches && cast.contains(document.activeElement) && document.activeElement !== button) expanded = true;
        update();
    });
    new MutationObserver(update).observe(document.getElementById('speaker-count'), {attributes:true,subtree:true,attributeFilter:['checked','aria-checked']});
    update();
}

// Keep the real textarea/caret untouched while measuring its wrapped text.
function writingStudioEditorHeight(natural, available, minimum = 72) {
    const cap = Math.max(minimum, Math.floor(Number.isFinite(available) ? available : minimum));
    return Math.min(cap, Math.max(minimum, Math.ceil(Number.isFinite(natural) ? natural : minimum)));
}
function initWritingStudioAutosize(editor, paint) {
    const grid = document.getElementById('composer-grid');
    const card = document.getElementById('composer-script');
    if (!grid || !card || !editor.cloneNode) return;
    const measure = editor.cloneNode(false);
    measure.removeAttribute('id'); measure.removeAttribute('name');
    measure.removeAttribute('autofocus'); measure.removeAttribute('required');
    measure.setAttribute('aria-hidden', 'true'); measure.tabIndex = -1;
    measure.style.cssText = 'position:fixed!important;left:-10000px!important;top:0!important;visibility:hidden!important;height:0!important;min-height:0!important;max-height:none!important;overflow:hidden!important;pointer-events:none!important;';
    document.body.append(measure);
    let pending = false;
    const resize = () => {
        pending = false;
        const style = getComputedStyle(editor);
        for (const key of ['fontFamily','fontSize','fontWeight','fontStyle','lineHeight','letterSpacing','wordSpacing','textIndent','whiteSpace','overflowWrap','wordBreak','paddingTop','paddingBottom','paddingLeft','paddingRight','borderTopWidth','borderBottomWidth','borderLeftWidth','borderRightWidth','boxSizing']) measure.style[key] = style[key];
        measure.style.width = editor.getBoundingClientRect().width + 'px';
        measure.value = editor.value || editor.placeholder || '';
        const border = (parseFloat(style.borderTopWidth) || 0) + (parseFloat(style.borderBottomWidth) || 0);
        const rect = editor.getBoundingClientRect();
        const footer = Math.max(0, card.getBoundingClientRect().bottom - rect.bottom);
        const available = grid.getBoundingClientRect().bottom - rect.top - footer - 2;
        const height = writingStudioEditorHeight(measure.scrollHeight + border, available);
        const value = height + 'px';
        if (card.style.getPropertyValue('--editor-height') !== value) card.style.setProperty('--editor-height', value);
        paint();
    };
    const schedule = () => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(resize);
    };
    editor.addEventListener('input', schedule);
    editor.addEventListener('change', schedule);
    window.addEventListener('resize', schedule);
    window.visualViewport?.addEventListener('resize', schedule);
    if (typeof ResizeObserver !== 'undefined') {
        const observer = new ResizeObserver(schedule);
        for (const item of [grid, card, editor, document.getElementById('composer-speakers'), document.getElementById('composer-transport')]) if (item) observer.observe(item);
    }
    if (typeof MutationObserver !== 'undefined') {
        const observer = new MutationObserver(schedule);
        for (const id of ['composer-speakers', 'script-check-feedback', 'generation-feedback']) {
            const item = document.getElementById(id);
            if (item) observer.observe(item, {childList:true, subtree:true, attributes:true, attributeFilter:['class','hidden','aria-expanded']});
        }
    }
    schedule();
}

// Voice identifiers are searchable names, not prose. Keep browser spelling aids
// from implying that a valid selected voice is an error. Never change its value.
function initVoiceInputPresentation() {
    const host = document.getElementById('voice-assignments');
    if (!host?.querySelectorAll) return;
    const apply = () => {
        for (const input of host.querySelectorAll('input')) {
            input.setAttribute('spellcheck', 'false');
            input.setAttribute('autocorrect', 'off');
            input.setAttribute('autocapitalize', 'none');
        }
    };
    // The chevron shares the input's native dropdown, without a second tab stop.
    host.addEventListener?.('pointerdown', event => {
        const arrow = event.target.closest?.('.icon-wrap');
        const input = arrow?.closest('.speaker-item')?.querySelector('input');
        if (!input || event.button !== 0) return;
        event.preventDefault(); event.stopPropagation();
        const open = input.getAttribute('aria-expanded') === 'true';
        input.focus();
        if (!open) input.dispatchEvent(new FocusEvent("focus"));
        input.dispatchEvent(new KeyboardEvent('keydown', {key:open ? 'Escape' : 'ArrowDown', code:open ? 'Escape' : 'ArrowDown', bubbles:true}));
    });
    host.addEventListener?.('keydown', event => {
        const input = event.target;
        if (input.matches?.('input[aria-controls="dropdown-options"]') && input.getAttribute('aria-expanded') === 'false' && ['ArrowDown', 'ArrowUp'].includes(event.key)) input.dispatchEvent(new FocusEvent('focus'));
    });
    apply();
    if (typeof MutationObserver !== 'undefined') new MutationObserver(apply).observe(host, {childList:true, subtree:true});
}

function initAudioLegend() {
    const dock = document.getElementById('composer-transport');
    if (!dock || document.getElementById('audio-section-label')) return;
    const label = document.createElement('h2');
    label.id = 'audio-section-label'; label.textContent = 'Audio';
    dock.prepend(label); dock.setAttribute('aria-labelledby', label.id);
}
