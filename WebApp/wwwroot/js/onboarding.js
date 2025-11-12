/**
 * Sistema de Onboarding y Tours para Ready4Hire
 */

class OnboardingService {
    constructor() {
        this.currentStep = 0;
        this.tours = new Map();
        this.activeTour = null;
        this.init();
    }

    init() {
        this.createOverlay();
    }

    createOverlay() {
        // Solo crear si no existe
        if (document.getElementById('tour-overlay')) return;

        const overlay = document.createElement('div');
        overlay.id = 'tour-overlay';
        overlay.className = 'tour-overlay';
        document.body.appendChild(overlay);

        const popover = document.createElement('div');
        popover.id = 'tour-popover';
        popover.className = 'tour-popover';
        document.body.appendChild(popover);
    }

    startTour(tourId, steps) {
        if (!steps || steps.length === 0) return;

        this.activeTour = { id: tourId, steps };
        this.tours.set(tourId, { steps, completed: false });
        this.currentStep = 0;

        // Preguntar al usuario si quiere iniciar
        this.showStartDialog(steps[0]);
    }

    showStartDialog(firstStep) {
        // Crear diálogo de inicio
        const dialog = document.createElement('div');
        dialog.className = 'tour-start-dialog';
        dialog.innerHTML = `
            <div class="tour-start-content">
                <h3>👋 Bienvenido al Tour Guiado</h3>
                <p>Descubre cómo usar Ready4Hire paso a paso</p>
                <div class="tour-start-actions">
                    <button class="tour-button tour-button-primary" data-action="start">Iniciar Tour</button>
                    <button class="tour-button tour-button-secondary" data-action="skip">Saltar</button>
                </div>
            </div>
        `;

        document.body.appendChild(dialog);

        // Event listeners
        dialog.querySelector('[data-action="start"]').addEventListener('click', () => {
            this.beginTour();
            dialog.remove();
        });

        dialog.querySelector('[data-action="skip"]').addEventListener('click', () => {
            this.markAsSkipped();
            dialog.remove();
        });
    }

    beginTour() {
        this.updateUI();
        this.showStep(this.currentStep);
    }

    showStep(stepIndex) {
        if (!this.activeTour || stepIndex >= this.activeTour.steps.length) {
            this.endTour(true);
            return;
        }

        const step = this.activeTour.steps[stepIndex];
        const element = document.querySelector(step.target);

        if (!element) {
            // Elemento no encontrado, pasar al siguiente
            this.nextStep();
            return;
        }

        // Mostrar popover
        this.positionPopover(element, step, stepIndex);
    }

    positionPopover(element, step, index) {
        const overlay = document.getElementById('tour-overlay');
        const popover = document.getElementById('tour-popover');

        // Calcular posición
        const rect = element.getBoundingClientRect();
        const scrollY = window.scrollY;

        // Determinar posición del popover
        let position = step.position || 'bottom';
        const positions = this.calculatePositions(rect, position);

        popover.innerHTML = `
            <div class="tour-content">
                <h3 class="tour-title">${step.title}</h3>
                <p class="tour-description">${step.description}</p>
                <div class="tour-steps">
                    ${this.activeTour.steps.map((_, i) => 
                        `<span class="tour-step-indicator ${i === index ? 'active' : ''}"></span>`
                    ).join('')}
                </div>
                <div class="tour-actions">
                    <button class="tour-button tour-button-secondary" data-action="skip">Saltar Tour</button>
                    ${index > 0 ? '<button class="tour-button tour-button-secondary" data-action="prev">Anterior</button>' : ''}
                    <button class="tour-button tour-button-primary" data-action="next">${index === this.activeTour.steps.length - 1 ? 'Finalizar' : 'Siguiente'}</button>
                </div>
            </div>
        `;

        popover.style.top = `${positions.top}px`;
        popover.style.left = `${positions.left}px`;
        popover.className = `tour-popover tour-popover-${position}`;

        // Event listeners
        popover.querySelector('[data-action="next"]').addEventListener('click', () => this.nextStep());
        popover.querySelector('[data-action="prev"]').addEventListener('click', () => this.prevStep());
        popover.querySelector('[data-action="skip"]').addEventListener('click', () => this.endTour(false));

        // Resaltar elemento
        this.highlightElement(element);
        overlay.classList.add('active');
    }

    calculatePositions(elementRect, position) {
        const margin = 20;
        let top = 0;
        let left = 0;

        switch (position) {
            case 'top':
                top = elementRect.top + window.scrollY - margin - 200;
                left = elementRect.left + (elementRect.width / 2) - 160;
                break;
            case 'bottom':
                top = elementRect.bottom + window.scrollY + margin;
                left = elementRect.left + (elementRect.width / 2) - 160;
                break;
            case 'left':
                top = elementRect.top + window.scrollY + (elementRect.height / 2) - 100;
                left = elementRect.left - 340 - margin;
                break;
            case 'right':
                top = elementRect.top + window.scrollY + (elementRect.height / 2) - 100;
                left = elementRect.right + margin;
                break;
        }

        // Asegurar que esté dentro de la ventana
        top = Math.max(20, Math.min(top, window.innerHeight + window.scrollY - 220));
        left = Math.max(20, Math.min(left, window.innerWidth - 340));

        return { top, left };
    }

    highlightElement(element) {
        // Agregar clase para resaltar
        element.classList.add('tour-highlighted');
        
        // Scroll a elemento si es necesario
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    nextStep() {
        document.querySelectorAll('.tour-highlighted').forEach(el => {
            el.classList.remove('tour-highlighted');
        });
        this.currentStep++;
        this.showStep(this.currentStep);
    }

    prevStep() {
        document.querySelectorAll('.tour-highlighted').forEach(el => {
            el.classList.remove('tour-highlighted');
        });
        this.currentStep--;
        this.showStep(this.currentStep);
    }

    endTour(completed) {
        if (completed) {
            this.markAsCompleted();
            this.showToast('🎉 Tour completado. ¡Bienvenido a Ready4Hire!', 'success');
        } else {
            this.markAsSkipped();
        }

        document.querySelectorAll('.tour-highlighted').forEach(el => {
            el.classList.remove('tour-highlighted');
        });

        document.getElementById('tour-overlay').classList.remove('active');
        document.getElementById('tour-popover').style.display = 'none';
        
        this.activeTour = null;
        this.currentStep = 0;
    }

    markAsCompleted() {
        if (this.activeTour) {
            const tour = this.tours.get(this.activeTour.id);
            if (tour) {
                tour.completed = true;
                this.saveTourState();
            }
        }
    }

    markAsSkipped() {
        if (this.activeTour) {
            const tour = this.tours.get(this.activeTour.id);
            if (tour) {
                tour.skipped = true;
                this.saveTourState();
            }
        }
    }

    saveTourState() {
        try {
            const states = {};
            this.tours.forEach((tour, id) => {
                states[id] = tour.completed || tour.skipped;
            });
            localStorage.setItem('onboarding-states', JSON.stringify(states));
        } catch (e) {
            console.error('Error saving tour state:', e);
        }
    }

    shouldShowTour(tourId) {
        try {
            const states = JSON.parse(localStorage.getItem('onboarding-states') || '{}');
            return !states[tourId];
        } catch (e) {
            return true;
        }
    }

    updateUI() {
        // Actualizar UI según el paso actual
        if (this.activeTour) {
            document.body.classList.add('tour-active');
        }
    }

    showToast(message, type) {
        if (window.Toast) {
            window.Toast.show(message, type);
        }
    }
}

// Inicializar servicio
window.OnboardingService = new OnboardingService();

// Tour inicial por defecto
const defaultTour = [
    {
        target: '.sidebar-container',
        title: 'Bienvenido a Ready4Hire',
        description: 'Este es tu panel de navegación principal. Desde aquí puedes acceder a todas las secciones.',
        position: 'right'
    },
    {
        target: '.nav-item[href="/home"]',
        title: 'Inicio',
        description: 'Ve a la página principal para ver tu progreso y estadísticas generales.',
        position: 'right'
    },
    {
        target: '.nav-item[href*="/chat"]',
        title: 'Entrevistas',
        description: 'Practica tus habilidades con entrevistas simuladas asistidas por IA.',
        position: 'right'
    },
    {
        target: '.nav-item[href="/gamification"]',
        title: 'Gamificación',
        description: 'Juega, gana puntos y desbloquea logros mientras aprendes.',
        position: 'right'
    }
];

// Agregar botón para reiniciar tour
function addTourButton() {
    const button = document.createElement('button');
    button.className = 'tour-restart-btn';
    button.setAttribute('aria-label', 'Reiniciar tour');
    button.innerHTML = 'ℹ️ Ayuda';
    button.addEventListener('click', () => {
        // Reiniciar tour
        localStorage.removeItem('onboarding-states');
        window.OnboardingService.startTour('default', defaultTour);
    });

    // Insertar en sidebar o header
    const sidebar = document.querySelector('.sidebar-footer');
    if (sidebar) {
        sidebar.appendChild(button);
    }
}

// Agregar estilos
const onboardingStyles = `
    .tour-start-dialog {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: var(--z-modal);
        animation: fadeIn 0.3s;
    }

    .tour-start-content {
        background: #1a1a1a;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-xl);
        padding: var(--space-8);
        max-width: 400px;
        text-align: center;
        animation: fadeInUp 0.3s;
    }

    .tour-start-content h3 {
        color: #ffffff;
        margin-bottom: var(--space-4);
        font-size: var(--font-2xl);
    }

    .tour-start-content p {
        color: #94a3b8;
        margin-bottom: var(--space-6);
    }

    .tour-start-actions {
        display: flex;
        gap: var(--space-4);
    }

    .tour-restart-btn {
        width: 100%;
        padding: var(--space-3) var(--space-4);
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-md);
        color: #94a3b8;
        cursor: pointer;
        transition: all 0.2s;
        margin-top: var(--space-2);
    }

    .tour-restart-btn:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff;
    }

    .tour-highlighted {
        position: relative;
        z-index: calc(var(--z-modal) + 1);
        outline: 2px solid var(--theme-primary);
        outline-offset: 4px;
    }

    body.tour-active {
        overflow: hidden;
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = onboardingStyles;
document.head.appendChild(styleSheet);

// Auto-iniciar tour si es la primera vez
document.addEventListener('DOMContentLoaded', () => {
    addTourButton();
    
    if (window.OnboardingService.shouldShowTour('default')) {
        setTimeout(() => {
            window.OnboardingService.startTour('default', defaultTour);
        }, 1000);
    }
});

