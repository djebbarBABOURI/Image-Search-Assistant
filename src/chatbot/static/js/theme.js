
// Thème clair / sombre : choisi avant le premier rendu pour éviter un flash.
(function () {
    var theme = null;

    try {
        theme = localStorage.getItem('theme');
    } catch (e) {
        /* Stockage indisponible. */
    }

    if (theme !== 'light' && theme !== 'dark') {
        theme = window.matchMedia &&
            window.matchMedia('(prefers-color-scheme: dark)').matches
            ? 'dark'
            : 'light';
    }

    document.documentElement.setAttribute('data-bs-theme', theme);
})();
