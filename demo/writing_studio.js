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
    initCastDisclosure();
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
