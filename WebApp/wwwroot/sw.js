/**
 * Service Worker para Ready4Hire
 * Habilita funcionalidad offline y caché inteligente
 */

const CACHE_NAME = 'ready4hire-v1';
const RUNTIME_CACHE = 'ready4hire-runtime';

// Recursos para caché estático
const STATIC_CACHE_URLS = [
    '/',
    '/css/improvements.css',
    '/css/main-professional.css',
    '/css/modern-sidebar.css',
    '/js/toast-notifications.js',
    '/js/theme-service.js',
    '/js/search-service.js'
];

// Instalación
self.addEventListener('install', (event) => {
    console.log('[Service Worker] Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[Service Worker] Caching static assets');
                return cache.addAll(STATIC_CACHE_URLS);
            })
            .then(() => self.skipWaiting())
    );
});

// Activación
self.addEventListener('activate', (event) => {
    console.log('[Service Worker] Activating...');
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME && cacheName !== RUNTIME_CACHE) {
                        console.log('[Service Worker] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Estrategia: Network first, fallback a cache
self.addEventListener('fetch', (event) => {
    // Solo interceptar requests GET
    if (event.request.method !== 'GET') return;

    // Estrategia para diferentes tipos de recursos
    if (event.request.destination === 'image') {
        // Imágenes: Cache first
        event.respondWith(cacheFirst(event.request));
    } else if (event.request.destination === 'script' || 
               event.request.destination === 'style' ||
               event.request.url.includes('/css/') ||
               event.request.url.includes('/js/')) {
        // CSS/JS: Cache first
        event.respondWith(cacheFirst(event.request));
    } else {
        // Otros: Network first
        event.respondWith(networkFirst(event.request));
    }
});

// Estrategia Cache First
async function cacheFirst(request) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    
    if (cached) {
        return cached;
    }
    
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        console.log('[Service Worker] Network failed for:', request.url);
        // Retornar respuesta genérica offline
        return new Response('Offline', { status: 503 });
    }
}

// Estrategia Network First
async function networkFirst(request) {
    const cache = await caches.open(RUNTIME_CACHE);
    
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        console.log('[Service Worker] Network failed, trying cache');
        const cached = await cache.match(request);
        if (cached) {
            return cached;
        }
        return new Response('Offline', { status: 503 });
    }
}

// Sincronización en background
self.addEventListener('sync', (event) => {
    console.log('[Service Worker] Background sync:', event.tag);
    // Implementar lógica de sincronización si es necesario
});

// Notificaciones push
self.addEventListener('push', (event) => {
    console.log('[Service Worker] Push notification received');
    const data = event.data ? event.data.json() : {};
    
    const options = {
        title: data.title || 'Ready4Hire',
        body: data.body || 'Tienes una notificación',
        icon: '/favicon.ico',
        badge: '/favicon.ico',
        vibrate: [200, 100, 200]
    };
    
    event.waitUntil(
        self.registration.showNotification(options.title, options)
    );
});

// Clik en notificación
self.addEventListener('notificationclick', (event) => {
    console.log('[Service Worker] Notification clicked');
    event.notification.close();
    
    event.waitUntil(
        clients.openWindow('/')
    );
});
