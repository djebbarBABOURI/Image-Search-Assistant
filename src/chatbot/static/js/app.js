/* Assistant Oxford Pets : client web avec gestion des images.
   API utilisée : GET /api/models, GET /api/tools, POST /api/upload, POST /api/chat. */
(() => {
    'use strict';

    /* Utilitaires */
    const $ = (selector) => document.querySelector(selector);
    const REDUCED_MOTION = window.matchMedia(
        '(prefers-reduced-motion: reduce)'
    ).matches;

    function h(tag, props = {}, ...kids) {
        const el = document.createElement(tag);

        for (const [key, val] of Object.entries(props)) {
            if (val == null || val === false) continue;

            if (key === 'class') {
                el.className = val;
            } else if (key.startsWith('on')) {
                el.addEventListener(key.slice(2), val);
            } else if (key === 'style' && typeof val === 'object') {
                Object.assign(el.style, val);
            } else {
                el.setAttribute(key, val === true ? '' : val);
            }
        }

        for (const kid of kids.flat()) {
            if (kid != null && kid !== false) el.append(kid);
        }

        return el;
    }

    const icon = (name) =>
        h('i', { class: `bi bi-${name}`, 'aria-hidden': 'true' });

    const pct = (p) =>
        `${(Number(p) * 100).toLocaleString('fr-FR', {
            maximumFractionDigits: 1
        })} %`;

    const num = (n) => Number(n ?? 0).toLocaleString('fr-FR');
    const pretty = (label) => String(label ?? '').replaceAll('_', ' ');

    const store = {
        get(key) {
            try {
                return localStorage.getItem(key);
            } catch {
                return null;
            }
        },
        set(key, value) {
            try {
                localStorage.setItem(key, value);
            } catch {
                /* Stockage indisponible. */
            }
        }
    };

    /* Éléments et état */
    const els = {
        chat: $('#chat'),
        messages: $('#messages'),
        welcome: $('#welcome'),
        form: $('#composer'),
        input: $('#input'),
        send: $('#send'),
        model: $('#model'),
        tools: $('#tools'),
        toasts: $('#toasts'),
        theme: $('#theme'),
        reset: $('#reset'),
        current: $('#current'),
        pending: $('#pending'),
        file: $('#file'),
        attach: $('#attach'),
        drop: $('#drop'),
        lightboxEl: $('#lightbox'),
        lightboxImg: $('#lightbox-img'),
        lightboxCap: $('#lightbox-cap')
    };

    const state = {
        history: [],
        busy: false,
        toolSeq: 0,
        image: null,
        lastImage: null
    };

    /* Notifications */
    function toast(message, kind = 'danger') {
        const element = h(
            'div',
            {
                class: `toast align-items-center text-bg-${kind} border-0`,
                role: 'alert',
                'aria-live': 'assertive',
                'aria-atomic': 'true'
            },
            h(
                'div',
                { class: 'd-flex' },
                h('div', { class: 'toast-body' }, message),
                h('button', {
                    type: 'button',
                    class: 'btn-close btn-close-white me-2 m-auto',
                    'data-bs-dismiss': 'toast',
                    'aria-label': 'Fermer'
                })
            )
        );

        els.toasts.append(element);

        const instance = new bootstrap.Toast(element, { delay: 6000 });
        element.addEventListener('hidden.bs.toast', () => element.remove());
        instance.show();
    }

    /* Thème clair / sombre */
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-bs-theme', theme);
        els.theme.replaceChildren(icon(theme === 'dark' ? 'sun' : 'moon-stars'));
        store.set('theme', theme);
    }

    els.theme.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-bs-theme');
        applyTheme(current === 'dark' ? 'light' : 'dark');
    });

    applyTheme(
        document.documentElement.getAttribute('data-bs-theme') || 'light'
    );

    /* Appels API */
    async function getJSON(url) {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Erreur ${response.status}`);
        }

        return response.json();
    }

    async function detail(response, fallback) {
        try {
            const data = await response.json();
            return data.detail || fallback;
        } catch {
            return fallback;
        }
    }

    /* Chargement des modèles Ollama */
    async function loadModels() {
        try {
            const data = await getJSON('/api/models');
            const models = data.models || [];

            if (!models.length) {
                els.model.replaceChildren(
                    h('option', { value: '' }, 'Aucun modèle (Ollama est-il lancé ?)')
                );
                return;
            }

            els.model.replaceChildren(
                ...models.map((model) => h('option', { value: model }, model))
            );

            const saved = store.get('model');

            if (saved && models.includes(saved)) {
                els.model.value = saved;
            } else if (data.default && models.includes(data.default)) {
                els.model.value = data.default;
            }
        } catch {
            els.model.replaceChildren(
                h('option', { value: '' }, 'Ollama injoignable')
            );
        }
    }

    els.model.addEventListener('change', () => {
        store.set('model', els.model.value);
    });

    /* Chargement des outils MCP */
    async function loadTools() {
        try {
            const tools = await getJSON('/api/tools');

            els.tools.replaceChildren(
                ...tools.map((tool) =>
                    h(
                        'span',
                        {
                            class: 'tool-pill',
                            title: tool.description || ''
                        },
                        tool.name
                    )
                )
            );
        } catch {
            els.tools.replaceChildren(
                h(
                    'span',
                    { class: 'text-body-secondary small' },
                    'Outils indisponibles : le serveur MCP est-il lancé ?'
                )
            );
        }
    }

    /* Lightbox et vignettes */
    const lightbox = new bootstrap.Modal(els.lightboxEl);

    function openLightbox(src, caption) {
        els.lightboxImg.src = src;
        els.lightboxCap.textContent = caption || '';
        lightbox.show();
    }

    const imgUrl = (item, size = 'thumb') => {
        if (item.url) return item.url;
        const id = item.image_id || item.id;
        return id ? `/api/image/${encodeURIComponent(id)}?size=${size}` : null;
    };

    function shot(item, { badge = null, title = '', sub = '', ref = false } = {}) {
        const src = imgUrl(item);
        const full = imgUrl(item, 'full');
        const content = src
            ? h('img', {
                src, alt: title, loading: 'lazy',
                onerror: (e) => e.target.replaceWith(
                    h('span', { class: 'shot-missing' }, icon('image'), 'Indisponible'))
            })
            : h('span', { class: 'shot-missing' }, icon('image'), 'Indisponible');

        return h('figure', { class: `shot${ref ? ' shot-ref' : ''}` },
            h('button', {
                type: 'button', class: 'shot-btn',
                'aria-label': `Agrandir ${title}`,
                onclick: () => full && openLightbox(full, `${title} ${sub}`.trim())
            },
                content,
                badge != null ? h('span', { class: 'shot-badge' }, String(badge)) : null),
            h('figcaption', {},
                h('span', { class: 'shot-title' }, title),
                sub ? h('span', { class: 'shot-sub' }, sub) : null));
    }

    const gallery = (items, mapper) =>
        h('div', { class: 'gallery' }, items.map(mapper));

    /* Rendu des résultats d'outils */
    const TOOL_LABELS = {
        classify: ['Classification', 'bullseye'],
        find_similar: ['Images proches', 'images'],
        filter_images: ['Filtrage', 'funnel'],
        count_by: ['Comptage', 'bar-chart'],
        top_confusions: ['Confusions', 'shuffle'],
        list_breeds: ['Liste des races', 'list-ul'],
        get_stats: ['Statistiques', 'graph-up'],
        get_metrics: ['Métriques du modèle', 'speedometer2']
    };

    const speciesTag = (species) =>
        h(
            'span',
            {
                class: `tag tag-${species === 'cat' ? 'cat' : 'dog'}`
            },
            species === 'cat' ? 'chat' : 'chien'
        );

    function bar(value, max = 1) {
        const width = max > 0
            ? Math.max(0, Math.min(100, (value / max) * 100))
            : 0;

        return h(
            'div',
            {
                class: 'progress',
                role: 'progressbar',
                'aria-valuemin': '0',
                'aria-valuemax': '100',
                'aria-valuenow': String(Math.round(width))
            },
            h('div', {
                class: 'progress-bar',
                style: `width:${width.toFixed(1)}%`
            })
        );
    }

    function barRow(name, valueText, value, max, extra = null) {
        return h(
            'div',
            { class: 'row-item' },
            h(
                'div',
                { class: 'row-top' },
                h('span', { class: 'row-name' }, name, extra),
                h('span', { class: 'row-val' }, valueText)
            ),
            bar(value, max)
        );
    }

    const RENDERERS = {
        count_by(data) {
            if (data.label !== undefined) {
                const suffix = data.min_confidence > 0
                    ? ` avec une confiance d’au moins ${pct(data.min_confidence)}`
                    : '';

                return h(
                    'div',
                    { class: 'stat' },
                    h('strong', {}, num(data.count)),
                    h('span', {}, `images prédites ${pretty(data.label)}${suffix}`)
                );
            }

            const entries = Object.entries(data.counts || {});
            const max = entries.length ? Math.max(...entries.map(([, n]) => n)) : 1;
            const limit = 10;

            return h(
                'div',
                {},
                h(
                    'div',
                    { class: 'rows' },
                    entries
                        .slice(0, limit)
                        .map(([label, count]) =>
                            barRow(pretty(label), num(count), count, max)
                        )
                ),
                h(
                    'p',
                    { class: 'note' },
                    entries.length > limit
                        ? `+ ${entries.length - limit} autres races. Total : ${num(data.total)} images.`
                        : `Total : ${num(data.total)} images.`
                )
            );
        },

        list_breeds(data) {
            return h(
                'div',
                { class: 'breeds' },
                (data.breeds || []).map((breed) =>
                    h(
                        'span',
                        { class: 'breed' },
                        pretty(breed.name),
                        speciesTag(breed.species)
                    )
                )
            );
        },

        classify(data) {
            const top = data.top || data.predictions || [];
            const ref = state.image
                ? shot(state.image, { title: 'Votre image', ref: true })
                : null;

            return h(
                'div',
                { class: 'classify' },
                ref,
                h(
                    'div',
                    { class: 'rows' },
                    top.map((p) => {
                        const conf = p.confidence ?? p.probability ?? 0;

                        return barRow(
                            pretty(p.breed || p.label),
                            pct(conf),
                            conf,
                            1,
                            p.species ? speciesTag(p.species) : null
                        );
                    })
                )
            );
        },

        find_similar(data) {
            const items = data.results || data.images || [];

            return gallery(items, (it, i) =>
                shot(it, {
                    badge: i + 1,
                    title: pretty(it.predicted_breed || it.breed),
                    sub: it.similarity != null ? pct(it.similarity) : ''
                })
            );
        },

        filter_images(data) {
            const items = data.images || data.results || [];

            return h(
                'div',
                {},
                gallery(items, (it) =>
                    shot(it, {
                        title: pretty(it.predicted_breed || it.breed),
                        sub: it.confidence != null ? pct(it.confidence) : ''
                    })
                ),
                data.total != null
                    ? h('p', { class: 'note' }, `${num(data.total)} images correspondantes.`)
                    : null
            );
        },

        top_confusions(data) {
            const confusions = data.confused_with || [];
            const max = Math.max(...confusions.map((c) => c.count), 1);

            return h(
                'div',
                {},
                h(
                    'div',
                    { class: 'stat' },
                    h('strong', {}, pct(data.error_rate)),
                    h(
                        'span',
                        {},
                        `d’erreur pour ${pretty(data.label)} (${num(data.misclassified)} sur ${num(data.total_images)})`
                    )
                ),
                confusions.length
                    ? h(
                        'div',
                        { class: 'confusions mt-3' },
                        confusions.map((c) =>
                            h(
                                'div',
                                { class: 'confusion' },
                                barRow(
                                    `Race prédite : ${pretty(c.predicted_breed)}`,
                                    `${num(c.count)} images`,
                                    c.count,
                                    max
                                ),
                                (c.examples || []).length
                                    ? gallery(c.examples, (it) =>
                                        shot(it, {
                                            title: pretty(it.predicted_breed || c.predicted_breed),
                                            sub: it.confidence != null ? pct(it.confidence) : ''
                                        })
                                    )
                                    : null
                            )
                        )
                    )
                    : h('p', { class: 'note' }, 'Aucune confusion signalée pour cette race.')
            );
        }
    };

    function renderResult(name, raw) {
        let data = null;

        try {
            data = JSON.parse(raw);
        } catch {
            /* Résultat en texte brut. */
        }

        if (data && typeof data === 'object' && data.error) {
            return h('div', { class: 'alert alert-warning py-2 mb-0' }, data.error);
        }

        if (data && RENDERERS[name]) {
            try {
                return RENDERERS[name](data);
            } catch {
                /* En cas de résultat inattendu, afficher le résultat brut. */
            }
        }

        return h('p', { class: 'plain' }, raw);
    }

    function formatArgs(args) {
        const parts = Object.entries(args || {}).map(([key, value]) => {
            const formatted = typeof value === 'string'
                ? value
                : JSON.stringify(value);

            return `${key}=${formatted}`;
        });

        return parts.length ? parts.join(', ') : 'sans argument';
    }

    function toolCard(tool) {
        const [label, iconName] = TOOL_LABELS[tool.name] || [
            tool.name,
            'wrench'
        ];

        const id = `tool-${++state.toolSeq}`;
        let raw = tool.result;

        try {
            raw = JSON.stringify(JSON.parse(tool.result), null, 2);
        } catch {
            /* Résultat en texte brut. */
        }

        return h(
            'div',
            { class: 'tool-card' },
            h(
                'button',
                {
                    type: 'button',
                    class: 'tool-head',
                    'data-bs-toggle': 'collapse',
                    'data-bs-target': `#${id}`,
                    'aria-expanded': 'true',
                    'aria-controls': id
                },
                icon(iconName),
                h('span', { class: 'tool-title' }, label),
                h(
                    'span',
                    { class: 'tool-code' },
                    `${tool.name}(${formatArgs(tool.args)})`
                ),
                icon('chevron-down')
            ),
            h(
                'div',
                { class: 'collapse show', id },
                h(
                    'div',
                    { class: 'tool-body' },
                    renderResult(tool.name, tool.result),
                    h(
                        'details',
                        { class: 'tool-raw' },
                        h('summary', {}, 'Détail brut'),
                        h(
                            'pre',
                            {},
                            `Arguments : ${JSON.stringify(tool.args)}\n\nRésultat :\n${raw}`
                        )
                    )
                )
            )
        );
    }

    /* Image jointe et image courante */
    async function uploadImage(file) {
        if (!file || !file.type.startsWith('image/')) {
            toast('Le fichier doit être une image.');
            return;
        }

        try {
            const response = await fetch('/api/upload', { method: 'POST', body: file });

            if (!response.ok) {
                throw new Error(await detail(response, `Erreur ${response.status}`));
            }

            const data = await response.json();

            state.image = {
                id: data.image_id,
                url: data.url || URL.createObjectURL(file),
                name: file.name
            };

            renderPending();
            renderCurrent(state.image);
        } catch (error) {
            toast(error.message || 'Échec du téléversement.');
        }
    }

    function renderPending() {
        const img = state.image;
        els.pending.hidden = !img;

        if (!img) {
            els.pending.replaceChildren();
            return;
        }

        els.pending.replaceChildren(
            h('img', { src: img.url, alt: '' }),
            h('span', {}, img.name, ' · ', h('code', {}, img.id)),
            h('button', {
                type: 'button',
                class: 'btn-close',
                'aria-label': 'Retirer l’image',
                onclick: () => {
                    state.image = null;
                    renderPending();
                }
            })
        );
    }

    function renderCurrent(img) {
        els.current.classList.toggle('has-image', !!img);
        els.current.replaceChildren(
            ...(img
                ? [h('img', { src: img.url, alt: img.name }), h('code', {}, img.id)]
                : ['Aucune image envoyée.'])
        );
    }

    /* Affichage des messages */
    function scrollToEnd() {
        els.chat.scrollTo({
            top: els.chat.scrollHeight,
            behavior: REDUCED_MOTION ? 'auto' : 'smooth'
        });
    }

    function markdown(text) {
        const box = h('div', { class: 'prose' });

        if (window.marked && window.DOMPurify) {
            box.innerHTML = DOMPurify.sanitize(
                marked.parse(text, { breaks: true }),
                { FORBID_TAGS: ['img', 'style'] }
            );

            box.querySelectorAll('table').forEach((table) => {
                table.classList.add('table', 'table-sm');

                const wrapper = h('div', { class: 'table-responsive' });
                table.replaceWith(wrapper);
                wrapper.append(table);
            });

            box.querySelectorAll('a').forEach((link) => {
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
            });
        } else {
            box.textContent = text;
        }

        return box;
    }

    function addUser(text, image = null) {
        els.messages.append(
            h(
                'div',
                { class: 'msg-user' },
                image ? h('img', { src: image.url, alt: image.name }) : null,
                text ? h('div', { class: 'bubble' }, text) : null
            )
        );

        scrollToEnd();
    }

    function addBot({
        text,
        tools = [],
        error = false,
        typing = false
    }) {
        const body = h('div', { class: 'msg-bot-body' });

        if (typing) {
            body.append(
                h(
                    'span',
                    {
                        class: 'typing',
                        role: 'status',
                        'aria-label': 'Réponse en cours'
                    },
                    h('i'),
                    h('i'),
                    h('i')
                )
            );
        } else {
            if (tools.length) {
                body.append(
                    h('div', { class: 'tool-stack' }, tools.map(toolCard))
                );
            }

            if (text) {
                body.append(
                    error
                        ? h('div', { class: 'prose' }, text)
                        : markdown(text)
                );
            }
        }

        const node = h(
            'div',
            { class: `msg-bot${error ? ' msg-error' : ''}` },
            h(
                'span',
                { class: 'avatar' },
                icon(error ? 'exclamation-triangle' : 'binoculars')
            ),
            body
        );

        els.messages.append(node);
        scrollToEnd();

        return node;
    }

    /* Gestion de la conversation */
    function setBusy(busy) {
        state.busy = busy;
        els.send.disabled = busy;
        els.send.setAttribute('aria-busy', String(busy));
    }

    async function submit(event) {
        event.preventDefault();

        if (state.busy) return;

        const text = els.input.value.trim();
        const image = state.image;
        if (!text && !image) return;

        els.welcome.hidden = true;
        addUser(text, image);

        state.history.push({
            role: 'user',
            content: (text || 'Quelle est la race de cet animal ?') +
                (image ? `\n[image_id: ${image.id}]` : '')
        });

        if (image) state.lastImage = image;

        els.input.value = '';
        els.input.style.height = 'auto';

        state.image = null;
        renderPending();
        renderCurrent(image);

        setBusy(true);
        const waiting = addBot({ typing: true });

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: state.history,
                    model: els.model.value
                })
            });

            if (!response.ok) {
                throw new Error(
                    await detail(response, `Erreur serveur ${response.status}.`)
                );
            }

            const data = await response.json();
            waiting.remove();

            const reply = data.reply || '(réponse vide)';

            addBot({
                text: reply,
                tools: data.tools || []
            });

            state.history.push({
                role: 'assistant',
                content: data.reply || ''
            });
        } catch (error) {
            waiting.remove();

            addBot({
                text: error.message || 'Erreur inattendue.',
                error: true
            });

            state.history.pop();
            state.image = image;
            renderPending();
        } finally {
            setBusy(false);
            els.input.focus();
        }
    }

    function resetConversation() {
        if (state.busy) return;

        state.history = [];
        state.image = null;
        renderPending();
        renderCurrent(null);

        els.messages.replaceChildren();
        els.welcome.hidden = false;
        els.chat.scrollTo({ top: 0 });
        els.input.focus();
    }

    /* Événements */
    els.form.addEventListener('submit', submit);
    els.reset.addEventListener('click', resetConversation);

    els.input.addEventListener('keydown', (event) => {
        if (
            event.key === 'Enter' &&
            !event.shiftKey &&
            !event.isComposing
        ) {
            event.preventDefault();
            els.form.requestSubmit();
        }
    });

    els.input.addEventListener('input', () => {
        els.input.style.height = 'auto';
        els.input.style.height =
            `${Math.min(els.input.scrollHeight, 160)}px`;
    });

    document.querySelectorAll('.chip').forEach((chip) => {
        chip.addEventListener('click', () => {
            if (state.busy) return;

            els.input.value = chip.textContent.trim();
            els.form.requestSubmit();
        });
    });

    els.attach.addEventListener('click', () => els.file.click());

    els.file.addEventListener('change', () => {
        uploadImage(els.file.files[0]);
        els.file.value = '';
    });

    let dragDepth = 0;
    const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');

    window.addEventListener('dragenter', (e) => {
        if (!hasFiles(e)) return;
        dragDepth++;
        els.drop.hidden = false;
    });

    window.addEventListener('dragleave', () => {
        if (--dragDepth <= 0) {
            dragDepth = 0;
            els.drop.hidden = true;
        }
    });

    window.addEventListener('dragover', (e) => {
        if (hasFiles(e)) e.preventDefault();
    });

    window.addEventListener('drop', (e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dragDepth = 0;
        els.drop.hidden = true;
        uploadImage(e.dataTransfer.files[0]);
    });

    els.input.addEventListener('paste', (e) => {
        const file = [...(e.clipboardData?.files || [])].find((f) =>
            f.type.startsWith('image/')
        );

        if (file) {
            e.preventDefault();
            uploadImage(file);
        }
    });

    /* Démarrage */
    renderCurrent(null);
    loadTools();
    loadModels();
    els.input.focus();
})();