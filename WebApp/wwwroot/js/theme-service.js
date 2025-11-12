/**
 * Servicio de gestión de temas (Dark/Light Mode)
 */

class ThemeService {
    constructor() {
        this.theme = this.getStoredTheme() || 'dark';
        this.init();
    }

    init() {
        // Aplicar tema guardado
        this.applyTheme(this.theme);
        
        // Escuchar cambios del sistema
        this.watchSystemTheme();
        
        // Crear toggle button si no existe
        this.createToggleButton();
    }

    getStoredTheme() {
        try {
            return localStorage.getItem('theme');
        } catch (e) {
            return null;
        }
    }

    setStoredTheme(theme) {
        try {
            localStorage.setItem('theme', theme);
        } catch (e) {
            console.error('Error saving theme:', e);
        }
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.theme = theme;
        this.setStoredTheme(theme);
        
        // Actualizar icono del toggle
        this.updateToggleIcon();
    }

    toggleTheme() {
        const newTheme = this.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
        
        // Notificar cambio
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: newTheme } }));
    }

    updateToggleIcon() {
        const icon = document.querySelector('.theme-toggle-icon');
        if (icon) {
            icon.textContent = this.theme === 'dark' ? '🌙' : '☀️';
        }
    }

    createToggleButton() {
        // Solo crear si no existe
        if (document.querySelector('.theme-toggle')) return;

        const button = document.createElement('button');
        button.className = 'theme-toggle';
        button.setAttribute('aria-label', 'Cambiar tema');
        button.innerHTML = '<span class="theme-toggle-icon">🌙</span>';
        button.addEventListener('click', () => this.toggleTheme());

        // Insertar en sidebar si existe
        const sidebar = document.querySelector('.sidebar-footer') || document.querySelector('.sidebar-nav');
        if (sidebar) {
            sidebar.insertBefore(button, sidebar.firstChild);
        } else {
            // Insertar en body si no hay sidebar
            document.body.appendChild(button);
            button.style.position = 'fixed';
            button.style.top = '20px';
            button.style.right = '20px';
            button.style.zIndex = '10001';
        }

        this.updateToggleIcon();
    }

    watchSystemTheme() {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
        
        mediaQuery.addEventListener('change', (e) => {
            // Solo aplicar si no hay tema guardado
            if (!this.getStoredTheme()) {
                this.applyTheme(e.matches ? 'light' : 'dark');
            }
        });
    }
}

// Inicializar servicio
window.ThemeService = new ThemeService();

