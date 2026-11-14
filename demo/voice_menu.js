function initVoiceMenu() {
    const assignments = document.getElementById("voice-assignments");
    const catalog = document.getElementById("voice-menu-catalog");
    const feedback = document.getElementById("voice-menu-feedback");
    const status = feedback.querySelector('[role="status"]');
    const undo = feedback.querySelector("button");
    const announcement = document.getElementById("voice-menu-announcement");
    const savedVoices = () => JSON.parse(catalog.querySelector('[data-saved-voice-catalog]')?.dataset.savedVoiceCatalog || '{}');
    const inputs = 'input[aria-controls="dropdown-options"]';
    const options = '[data-testid="dropdown-option"]';
    const icons = {
        more: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>',
        trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"/></svg>',
    };
    let armed;
    let request;

    function paint() {
        const voices = savedVoices();
        if (armed && voices[armed.name] !== armed.revision) {
            armed = undefined;
            announcement.textContent = "";
        }
        for (const input of assignments.querySelectorAll(inputs)) {
            input.setAttribute("aria-describedby", "voice-menu-help");
            input.placeholder = "Choose a voice";
        }
        for (const row of assignments.querySelectorAll(options)) {
            const name = row.getAttribute("aria-label");
            const item = row.closest(".speaker-item");
            const open = item.querySelector(inputs)?.getAttribute("aria-expanded") === "true";
            let button = row.querySelector(".voice-row-action");
            if (!open || !Object.hasOwn(voices, name)) {
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
    observer.observe(catalog, {subtree: true, childList: true, attributes: true, attributeFilter: ["data-saved-voice-catalog", "data-update"]});
    paint();
    initPresetGroups(assignments, catalog);

    function arm(row) {
        const name = row.getAttribute("aria-label");
        const revision = savedVoices()[name];
        if (!revision) return;
        armed = {item: row.closest(".speaker-item"), name, revision};
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
        request = {previousUpdate: document.querySelector("[data-voice-result]")?.dataset.update, input, reopen: input?.getAttribute("aria-expanded") === "true" || command === "undo"};
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
        if (button.dataset.action === "trash" && armed) {
            submit(`delete:${armed.revision}:${armed.name}`, armed.item.querySelector(inputs));
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
            submit(`delete:${armed.revision}:${armed.name}`, input);
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
        if (payload && voiceMenuResultIsNew(resultField.querySelector("[data-voice-result]")?.dataset.update, request?.previousUpdate)) window.vibevoiceVoiceMenu.finish(JSON.parse(payload));
    }).observe(resultField, {subtree: true, childList: true, attributes: true, attributeFilter: ["data-voice-result", "data-update"]});
}

// Native Gradio options retain their identity and selection handlers.
function initPresetGroups(assignments, catalog) {
 const states=new WeakMap();
 const selector='input[aria-controls="dropdown-options"]';
 function paint(){
  const names=new Set(JSON.parse(catalog.querySelector('[data-builtin-voice-catalog]')?.dataset.builtinVoiceCatalog||'[]'));
  for(const item of assignments.querySelectorAll('.speaker-item')){
   const input=item.querySelector(selector),list=item.querySelector('ul[role="listbox"]');
   let button=item.querySelector('.preset-disclosure');
   if(!list||input?.getAttribute('aria-expanded')!=='true'){button?.remove();states.delete(item);continue;}
   let state=states.get(item);if(!state){state={list,expanded:false};states.set(item,state);}else state.list=list;
   const rows=[...list.querySelectorAll('[data-testid="dropdown-option"]')];
   if(!rows.some(r=>names.has(r.getAttribute('aria-label')))){button?.remove();list.classList.remove('has-preset-disclosure');continue;}
   if(!button){
    button=document.createElement('button');button.type='button';button.className='preset-disclosure';button.tabIndex=-1;
    button.addEventListener('mousedown',e=>e.preventDefault());
    button.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();const current=states.get(item);if(current)current.expanded=!current.expanded;paint();});list.after(button);
   }
   list.classList.add('has-preset-disclosure');
   if(!list.id)list.id=item.id+'-voice-options';button.setAttribute('aria-controls',list.id);
   const text='Built-in presets '+(state.expanded?'−':'+');if(button.textContent!==text)button.textContent=text;
   if(button.getAttribute('aria-expanded')!==String(state.expanded))button.setAttribute('aria-expanded',String(state.expanded));
   button.title='Built-in presets. Right arrow expands; left arrow collapses.';
   const r=list.getBoundingClientRect();button.style.cssText=`position:fixed;left:${r.left+6}px;top:${r.top+5}px;width:${r.width-12}px;`;
   for(const row of rows){const preset=names.has(row.getAttribute('aria-label'));row.classList.toggle('builtin-preset',preset);row.hidden=preset&&!state.expanded;}
   input.setAttribute('aria-description','Right arrow expands Built-in presets; left arrow collapses. Typing or arrow navigation searches all voices.');
  }
 }
 assignments.addEventListener('keydown',e=>{
  if(!e.target.matches(selector)||e.target.getAttribute('aria-expanded')!=='true')return;const state=states.get(e.target.closest('.speaker-item'));if(!state)return;
  if(['ArrowRight','ArrowLeft'].includes(e.key)){state.expanded=e.key==='ArrowRight';e.preventDefault();e.stopImmediatePropagation();paint();}
  else if(['ArrowDown','ArrowUp','Home','End','Enter'].includes(e.key)&&!state.expanded){state.expanded=true;paint();if(e.key==='Enter'){e.preventDefault();e.stopImmediatePropagation();}}
 },true);
 assignments.addEventListener('input',e=>{const state=states.get(e.target.closest('.speaker-item'));if(state){state.expanded=true;paint();}});
 new MutationObserver(paint).observe(assignments,{subtree:true,childList:true,attributes:true,attributeFilter:['aria-expanded']});
 new MutationObserver(paint).observe(catalog,{subtree:true,childList:true,attributes:true,attributeFilter:['data-builtin-voice-catalog','data-update']});
 window.addEventListener('resize',paint);window.addEventListener('scroll',paint,true);
}

function voiceMenuResultIsNew(update, previous) { return Boolean(update) && update !== previous; }
