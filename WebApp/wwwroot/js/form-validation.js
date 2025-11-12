/**
 * Sistema de validación de formularios en tiempo real
 */

class FormValidation {
    constructor(form) {
        this.form = form;
        this.rules = {};
        this.errors = {};
        this.init();
    }

    init() {
        // Configurar validación en inputs
        const inputs = this.form.querySelectorAll('input, textarea, select');
        
        inputs.forEach(input => {
            // Validación en tiempo real
            input.addEventListener('blur', () => this.validateField(input));
            input.addEventListener('input', () => this.clearError(input));
            
            // Leer atributos de validación HTML5
            this.parseValidationAttributes(input);
        });

        // Validar en submit
        this.form.addEventListener('submit', (e) => {
            if (!this.validateForm()) {
                e.preventDefault();
            }
        });
    }

    parseValidationAttributes(input) {
        const name = input.getAttribute('name');
        if (!name) return;

        const rules = [];

        // Required
        if (input.hasAttribute('required')) {
            rules.push({
                type: 'required',
                message: 'Este campo es obligatorio'
            });
        }

        // Type validations
        const type = input.getAttribute('type');
        if (type === 'email') {
            rules.push({
                type: 'email',
                message: 'Email inválido',
                pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/
            });
        }

        if (type === 'password') {
            const minLength = input.getAttribute('minlength') || 8;
            rules.push({
                type: 'minlength',
                message: `La contraseña debe tener al menos ${minLength} caracteres`,
                minLength: parseInt(minLength)
            });
        }

        // Pattern
        const pattern = input.getAttribute('pattern');
        if (pattern) {
            rules.push({
                type: 'pattern',
                message: 'El formato no es válido',
                pattern: new RegExp(pattern)
            });
        }

        this.rules[name] = rules;
    }

    validateField(field) {
        const name = field.getAttribute('name');
        if (!name || !this.rules[name]) return true;

        const rules = this.rules[name];
        let isValid = true;

        for (const rule of rules) {
            if (!this.checkRule(field, rule)) {
                this.showError(field, rule.message);
                isValid = false;
                break;
            }
        }

        if (isValid) {
            this.clearError(field);
        }

        return isValid;
    }

    checkRule(field, rule) {
        const value = field.value.trim();

        switch (rule.type) {
            case 'required':
                return value !== '';
            
            case 'email':
                return rule.pattern.test(value);
            
            case 'minlength':
                return value.length >= rule.minLength;
            
            case 'pattern':
                return rule.pattern.test(value);
            
            default:
                return true;
        }
    }

    showError(field, message) {
        this.clearError(field);

        field.classList.add('invalid');
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.setAttribute('role', 'alert');
        errorDiv.textContent = message;
        
        field.parentNode.insertBefore(errorDiv, field.nextSibling);
        
        // Agregar aria-invalid
        field.setAttribute('aria-invalid', 'true');
        field.setAttribute('aria-describedby', errorDiv.id || '');
    }

    clearError(field) {
        field.classList.remove('invalid');
        
        const errorDiv = field.parentNode.querySelector('.field-error');
        if (errorDiv) {
            errorDiv.remove();
        }
        
        field.removeAttribute('aria-invalid');
        field.removeAttribute('aria-describedby');
    }

    validateForm() {
        let isValid = true;
        const inputs = this.form.querySelectorAll('input, textarea, select');

        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });

        if (!isValid) {
            this.showToast('Por favor corrige los errores en el formulario', 'error');
        }

        return isValid;
    }

    showToast(message, type) {
        if (window.Toast) {
            window.Toast.show(message, type);
        }
    }
}

// Inicializar validación para todos los formularios
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('form').forEach(form => {
        new FormValidation(form);
    });
});

// Estilos CSS para validación
const validationStyles = `
    .invalid {
        border-color: #ef4444 !important;
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1) !important;
    }

    .field-error {
        color: #ef4444;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        animation: shake 0.3s ease-out;
    }

    .field-error::before {
        content: '⚠';
        font-size: 1.2em;
    }

    .valid {
        border-color: #10b981 !important;
    }

    .valid::after {
        content: '✓';
        position: absolute;
        right: 10px;
        color: #10b981;
        font-weight: bold;
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = validationStyles;
document.head.appendChild(styleSheet);

