
/* Assistant Oxford Pets : client web sans gestion des images.
   API utilisée : GET /api/models, GET /api/tools, POST /api/chat. */
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
        reset: $('#reset')
    };

    const state = {
        history: [],
        busy: false,
        toolSeq: 0
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
                    h(
                        'span',
                        {},
                        `images prédites ${pretty(data.label)}${suffix}`
                    )
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

        top_confusions(data) {
            const confusions = data.confused_with || [];

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
                        { class: 'rows mt-3' },
                        confusions.map((item) =>
                            barRow(
                                `Race prédite : ${pretty(item.predicted_breed)}`,
                                `${num(item.count)} images`,
                                item.count,
                                Math.max(...confusions.map((entry) => entry.count), 1)
                            )
                        )
                    )
                    : h(
                        'p',
                        { class: 'note' },
                        'Aucune confusion signalée pour cette race.'
                    ),
                confusions.length
                    ? h(
                        'p',
                        { class: 'note' },
                        'Les résultats détaillés des exemples ne sont pas affichés dans cette version sans images.'
                    )
                    : null
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

    function addUser(text) {
        const node = h(
            'div',
            { class: 'msg-user' },
            h('div', { class: 'bubble' }, text)
        );

        els.messages.append(node);
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
        if (!text) return;

        els.welcome.hidden = true;
        addUser(text);

        state.history.push({
            role: 'user',
            content: text
        });

        els.input.value = '';
        els.input.style.height = 'auto';

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
        } finally {
            setBusy(false);
            els.input.focus();
        }
    }

    function resetConversation() {
        if (state.busy) return;

        state.history = [];
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

    /* Démarrage */
    loadTools();
    loadModels();
    els.input.focus();
})();