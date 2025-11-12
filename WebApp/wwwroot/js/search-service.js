/**
 * Sistema de búsqueda global para Ready4Hire
 */

class SearchService {
    constructor() {
        this.index = new Map();
        this.init();
    }

    init() {
        this.buildIndex();
    }

    buildIndex() {
        // Indexar elementos de navegación
        document.querySelectorAll('.nav-item').forEach(item => {
            const text = item.textContent.trim();
            const link = item.getAttribute('href');
            if (text && link) {
                this.index.set(text.toLowerCase(), {
                    text: text,
                    link: link,
                    type: 'navigation'
                });
            }
        });

        // Indexar contenido de páginas (títulos, encabezados)
        document.querySelectorAll('h1, h2, h3, [role="heading"]').forEach(heading => {
            const text = heading.textContent.trim();
            if (text) {
                this.index.set(text.toLowerCase(), {
                    text: text,
                    element: heading,
                    type: 'heading'
                });
            }
        });

        // Indexar botones principales
        document.querySelectorAll('.btn-primary, .btn-enhanced-primary').forEach(btn => {
            const text = btn.textContent.trim();
            if (text) {
                this.index.set(text.toLowerCase(), {
                    text: text,
                    element: btn,
                    type: 'action'
                });
            }
        });
    }

    search(query) {
        if (!query || query.length < 2) return [];

        const results = [];
        const lowerQuery = query.toLowerCase();

        // Búsqueda exacta
        if (this.index.has(lowerQuery)) {
            results.push(this.index.get(lowerQuery));
        }

        // Búsqueda parcial
        this.index.forEach((value, key) => {
            if (key.includes(lowerQuery) && !results.includes(value)) {
                results.push(value);
            }
        });

        return results.slice(0, 10); // Limitar a 10 resultados
    }

    highlight(element, query) {
        if (!element || !query) return;

        const text = element.textContent;
        const regex = new RegExp(`(${query})`, 'gi');
        const highlighted = text.replace(regex, '<mark>$1</mark>');
        element.innerHTML = highlighted;
    }

    removeHighlight(element) {
        if (!element) return;
        const text = element.textContent;
        element.textContent = text;
    }
}

// Inicializar servicio
window.SearchService = new SearchService();

// Función global para búsqueda rápida
window.quickSearch = function(query) {
    return window.SearchService.search(query);
};

