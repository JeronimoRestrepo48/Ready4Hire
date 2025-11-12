/**
 * Scroll Reveal Animation System
 */

class ScrollReveal {
    constructor() {
        this.elements = [];
        this.observer = null;
        this.init();
    }

    init() {
        if ('IntersectionObserver' in window) {
            this.observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.reveal(entry.target);
                    }
                });
            }, {
                threshold: 0.1,
                rootMargin: '0px 0px -100px 0px'
            });

            this.observeElements();
        } else {
            // Fallback: mostrar todo
            document.querySelectorAll('.reveal').forEach(el => {
                el.classList.add('active');
            });
        }
    }

    observeElements() {
        // Observar todos los elementos con clase .reveal
        document.querySelectorAll('.reveal').forEach(el => {
            this.observer.observe(el);
        });

        // Observar elements con data-reveal
        document.querySelectorAll('[data-reveal]').forEach(el => {
            const animation = el.dataset.reveal;
            el.classList.add('reveal', `reveal-${animation}`);
            this.observer.observe(el);
        });
    }

    reveal(element) {
        element.classList.add('active');
        
        // Agregar delay si está especificado
        const delay = element.dataset.revealDelay || 0;
        if (delay > 0) {
            setTimeout(() => {
                element.style.transitionDelay = `${delay}ms`;
            }, 50);
        }

        // No observar más este elemento
        if (this.observer) {
            this.observer.unobserve(element);
        }
    }

    reset() {
        // Remover todas las animaciones
        document.querySelectorAll('.reveal').forEach(el => {
            el.classList.remove('active');
            if (this.observer) {
                this.observer.observe(el);
            }
        });
    }
}

// Inicializar
window.ScrollRevealService = new ScrollReveal();

// Agregar estilos para animaciones de reveal
const revealStyles = `
    .reveal,
    [data-reveal] {
        opacity: 0;
        transition: opacity 0.6s ease-out, transform 0.6s ease-out;
    }

    .reveal-top {
        transform: translateY(30px);
    }

    .reveal-bottom {
        transform: translateY(-30px);
    }

    .reveal-left {
        transform: translateX(-30px);
    }

    .reveal-right {
        transform: translateX(30px);
    }

    .reveal-zoom {
        transform: scale(0.9);
    }

    .reveal.active {
        opacity: 1;
        transform: translate(0) scale(1);
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = revealStyles;
document.head.appendChild(styleSheet);

