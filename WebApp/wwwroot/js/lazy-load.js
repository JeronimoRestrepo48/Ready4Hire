/**
 * Sistema de Lazy Loading para Ready4Hire
 */

class LazyLoadService {
    constructor() {
        this.observer = null;
        this.init();
    }

    init() {
        // Intersection Observer para lazy loading
        if ('IntersectionObserver' in window) {
            this.observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.loadElement(entry.target);
                    }
                });
            }, {
                rootMargin: '50px',
                threshold: 0.01
            });

            this.observeImages();
            this.observeComponents();
        } else {
            // Fallback para navegadores antiguos
            this.loadAll();
        }
    }

    observeImages() {
        const images = document.querySelectorAll('img[data-src]');
        images.forEach(img => {
            this.observer.observe(img);
        });
    }

    observeComponents() {
        const components = document.querySelectorAll('[data-lazy-component]');
        components.forEach(component => {
            this.observer.observe(component);
        });
    }

    loadElement(element) {
        if (element.tagName === 'IMG' && element.dataset.src) {
            // Cargar imagen
            element.src = element.dataset.src;
            element.classList.add('lazy-loaded');
            
            // Remover atributo data-src para evitar recargar
            element.removeAttribute('data-src');
            
            // Event handler para error
            element.addEventListener('error', () => {
                this.handleImageError(element);
            });

            this.observer.unobserve(element);
        } else if (element.dataset.lazyComponent) {
            // Cargar componente
            this.loadComponent(element);
        }
    }

    handleImageError(img) {
        img.src = '/img/placeholder.png';
        img.alt = 'Imagen no disponible';
        img.classList.add('error');
    }

    loadComponent(component) {
        const componentUrl = component.dataset.lazyComponent;
        
        // Mostrar skeleton mientras carga
        component.classList.add('loading');
        
        fetch(componentUrl)
            .then(response => response.text())
            .then(html => {
                component.innerHTML = html;
                component.classList.remove('loading');
                component.classList.add('lazy-loaded');
                
                // Inicializar scripts del componente si es necesario
                this.initComponentScripts(component);
            })
            .catch(error => {
                console.error('Error loading component:', error);
                component.classList.remove('loading');
                component.classList.add('error');
            });

        this.observer.unobserve(component);
    }

    initComponentScripts(component) {
        // Buscar y ejecutar scripts dentro del componente
        const scripts = component.querySelectorAll('script');
        scripts.forEach(oldScript => {
            const newScript = document.createElement('script');
            Array.from(oldScript.attributes).forEach(attr => {
                newScript.setAttribute(attr.name, attr.value);
            });
            newScript.textContent = oldScript.textContent;
            oldScript.parentNode.replaceChild(newScript, oldScript);
        });
    }

    loadAll() {
        // Cargar todo inmediatamente si no hay IntersectionObserver
        document.querySelectorAll('img[data-src]').forEach(img => {
            img.src = img.dataset.src;
        });

        document.querySelectorAll('[data-lazy-component]').forEach(component => {
            this.loadComponent(component);
        });
    }
}

// Inicializar servicio
window.LazyLoadService = new LazyLoadService();

// Utilitario para agregar skeleton loader
function addSkeletonLoader(container) {
    container.innerHTML = `
        <div class="skeleton" style="height: 200px; margin-bottom: 16px;"></div>
        <div class="skeleton" style="height: 24px; margin-bottom: 8px;"></div>
        <div class="skeleton" style="height: 24px; width: 80%;"></div>
    `;
}

