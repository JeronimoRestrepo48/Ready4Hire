/**
 * Mejoras para el sistema de chat
 * Incluye indicadores de escritura, autoscroll suave, etc.
 */

class ChatEnhancements {
    constructor() {
        this.typingIndicators = new Map();
        this.lastScroll = 0;
        this.init();
    }

    init() {
        this.setupTypingIndicators();
        this.setupSmoothScroll();
        this.setupMessageAnimations();
    }

    // Indicadores de escritura
    setupTypingIndicators() {
        const chatInput = document.querySelector('.chat-input, [name="message"], textarea');
        
        if (chatInput) {
            let typingTimeout;
            
            chatInput.addEventListener('input', () => {
                this.showTypingIndicator();
                
                clearTimeout(typingTimeout);
                typingTimeout = setTimeout(() => {
                    this.hideTypingIndicator();
                }, 1000);
            });
        }
    }

    showTypingIndicator() {
        const chatMessages = document.querySelector('.chat-messages-container, .chat-messages');
        if (!chatMessages) return;

        const indicator = this.createTypingIndicator();
        chatMessages.appendChild(indicator);
        this.scrollToBottom(chatMessages);
    }

    hideTypingIndicator() {
        const indicator = document.querySelector('.typing-indicator');
        if (indicator) {
            indicator.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => indicator.remove(), 300);
        }
    }

    createTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'typing-indicator';
        div.setAttribute('aria-live', 'polite');
        div.innerHTML = `
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        return div;
    }

    // Scroll suave automático
    setupSmoothScroll() {
        const chatMessages = document.querySelector('.chat-messages-container, .chat-messages');
        if (!chatMessages) return;

        // Auto-scroll al agregar mensajes - siempre hacer scroll cuando se agregan mensajes nuevos
        const observer = new MutationObserver((mutations) => {
            // Verificar si se agregaron nuevos nodos
            const hasNewMessages = mutations.some(mutation => 
                mutation.addedNodes.length > 0 && 
                Array.from(mutation.addedNodes).some(node => 
                    node.nodeType === 1 && (node.classList.contains('message-wrapper') || node.querySelector('.message-wrapper'))
                )
            );
            
            if (hasNewMessages) {
                // Pequeño delay para asegurar que el DOM se actualizó completamente
                setTimeout(() => {
                this.scrollToBottom(chatMessages);
                }, 100);
            }
        });

        observer.observe(chatMessages, {
            childList: true,
            subtree: true
        });

        // Mantener scroll al cargar mensajes
        chatMessages.addEventListener('scroll', () => {
            this.lastScroll = chatMessages.scrollTop;
        });
        
        // También hacer scroll cuando cambia el contenido
        chatMessages.addEventListener('DOMSubtreeModified', () => {
            setTimeout(() => {
                this.scrollToBottom(chatMessages);
            }, 100);
        });
    }

    isNearBottom(element, threshold = 100) {
        return element.scrollHeight - element.scrollTop - element.clientHeight < threshold;
    }

    scrollToBottom(element) {
        if (!element) return;
        
        element.scrollTo({
            top: element.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Animaciones de mensajes
    setupMessageAnimations() {
        const messages = document.querySelectorAll('.message, .chat-message');
        
        messages.forEach((message, index) => {
            message.style.opacity = '0';
            message.style.animation = `fadeInUp 0.4s ease-out ${index * 0.05}s forwards`;
        });
    }

    // Agregar animación a nuevo mensaje
    animateNewMessage(element) {
        if (!element) return;
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        
        requestAnimationFrame(() => {
            element.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        });
    }

    // Quick replies
    addQuickReplies(replies) {
        const chatInput = document.querySelector('.chat-input-container, .chat-input-wrapper');
        if (!chatInput) return;

        const quickRepliesContainer = document.createElement('div');
        quickRepliesContainer.className = 'quick-replies';
        
        replies.forEach(reply => {
            const button = document.createElement('button');
            button.className = 'quick-reply-btn';
            button.textContent = reply;
            button.addEventListener('click', () => {
                this.sendQuickReply(reply);
            });
            quickRepliesContainer.appendChild(button);
        });

        chatInput.insertBefore(quickRepliesContainer, chatInput.firstChild);
    }

    sendQuickReply(reply) {
        const chatInput = document.querySelector('.chat-input, [name="message"], textarea');
        if (chatInput) {
            chatInput.value = reply;
            // Disparar evento de cambio para que el componente reaccione
            chatInput.dispatchEvent(new Event('change', { bubbles: true }));
            
            // Auto-enviar o esperar confirmación
            const sendButton = document.querySelector('.chat-send-button, [type="submit"]');
            if (sendButton) {
                setTimeout(() => sendButton.click(), 100);
            }
        }
    }
}

// Inicializar mejoras de chat
if (document.querySelector('.chat-container, .chat-messages')) {
    window.ChatEnhancements = new ChatEnhancements();
}

// Estilos CSS para los componentes
const chatEnhancementsStyles = `
    .typing-indicator {
        padding: 12px 16px;
        display: flex;
        align-items: center;
    }

    .typing-dots {
        display: flex;
        gap: 4px;
    }

    .typing-dots span {
        width: 8px;
        height: 8px;
        background: #94a3b8;
        border-radius: 50%;
        animation: typingDot 1.4s infinite ease-in-out;
    }

    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes typingDot {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
        30% { transform: translateY(-10px); opacity: 1; }
    }

    .quick-replies {
        display: flex;
        gap: 8px;
        padding: 12px 16px;
        overflow-x: auto;
        scrollbar-width: none;
    }

    .quick-replies::-webkit-scrollbar {
        display: none;
    }

    .quick-reply-btn {
        padding: 8px 16px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        color: #e2e8f0;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
        flex-shrink: 0;
    }

    .quick-reply-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
`;

// Inyectar estilos
const styleSheet = document.createElement('style');
styleSheet.textContent = chatEnhancementsStyles;
document.head.appendChild(styleSheet);

