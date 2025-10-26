// ══════════════════════════════════════════════════════════════════════════
// Service Worker para Ready4Hire PWA
// Características: Offline support, cache strategy, background sync
// ══════════════════════════════════════════════════════════════════════════

const CACHE_NAME = 'ready4hire-v2.1.0';
const RUNTIME_CACHE = 'ready4hire-runtime';
const API_CACHE = 'ready4hire-api';

// Archivos para pre-cachear (app shell)
const urlsToCache = [
  '/',
  '/css/app.css',
  '/css/modern-sidebar.css',
  '/css/modern-chat.css',
  '/_framework/blazor.web.js',
  '/manifest.json',
  '/images/icons/icon-192x192.png',
  '/images/icons/icon-512x512.png'
];

// ──────────────────────────────────────────────────────────────────────────
// INSTALL EVENT - Pre-cache app shell
// ──────────────────────────────────────────────────────────────────────────
self.addEventListener('install', event => {
  console.log('[SW] Installing Service Worker...', CACHE_NAME);
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[SW] Caching app shell');
        return cache.addAll(urlsToCache);
      })
      .then(() => self.skipWaiting()) // Activar inmediatamente
  );
});

// ──────────────────────────────────────────────────────────────────────────
// ACTIVATE EVENT - Limpiar caches viejos
// ──────────────────────────────────────────────────────────────────────────
self.addEventListener('activate', event => {
  console.log('[SW] Activating Service Worker...', CACHE_NAME);
  
  const cacheWhitelist = [CACHE_NAME, RUNTIME_CACHE, API_CACHE];
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (!cacheWhitelist.includes(cacheName)) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => self.clients.claim()) // Tomar control inmediatamente
  );
});

// ──────────────────────────────────────────────────────────────────────────
// FETCH EVENT - Cache strategies
// ──────────────────────────────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Estrategia 1: API Calls - Network First (with cache fallback)
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(request, API_CACHE));
    return;
  }
  
  // Estrategia 2: Blazor Framework - Cache First
  if (request.url.includes('/_framework/')) {
    event.respondWith(cacheFirstStrategy(request, CACHE_NAME));
    return;
  }
  
  // Estrategia 3: Imágenes y assets - Cache First
  if (request.destination === 'image' || request.destination === 'style' || request.destination === 'script') {
    event.respondWith(cacheFirstStrategy(request, RUNTIME_CACHE));
    return;
  }
  
  // Estrategia 4: HTML/Navegación - Network First
  if (request.mode === 'navigate') {
    event.respondWith(networkFirstStrategy(request, RUNTIME_CACHE));
    return;
  }
  
  // Default: Network First
  event.respondWith(networkFirstStrategy(request, RUNTIME_CACHE));
});

// ──────────────────────────────────────────────────────────────────────────
// CACHE STRATEGIES
// ──────────────────────────────────────────────────────────────────────────

/**
 * Cache First Strategy: Usa cache primero, network como fallback
 * Ideal para: Assets estáticos, imágenes, CSS, JS
 */
async function cacheFirstStrategy(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  
  if (cached) {
    console.log('[SW] Cache HIT:', request.url);
    return cached;
  }
  
  console.log('[SW] Cache MISS, fetching:', request.url);
  try {
    const response = await fetch(request);
    
    // Cachear la respuesta para futuras requests
    if (response.status === 200) {
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    console.error('[SW] Fetch failed:', error);
    // Retornar página offline si existe
    return cache.match('/offline.html') || new Response('Offline', { status: 503 });
  }
}

/**
 * Network First Strategy: Intenta network primero, cache como fallback
 * Ideal para: API calls, contenido dinámico
 */
async function networkFirstStrategy(request, cacheName) {
  const cache = await caches.open(cacheName);
  
  try {
    const response = await fetch(request);
    
    // Cachear respuesta exitosa
    if (response.status === 200) {
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    console.warn('[SW] Network failed, trying cache:', request.url);
    
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    
    // Si es API call fallido, retornar error JSON
    if (request.url.includes('/api/')) {
      return new Response(
        JSON.stringify({ error: 'Offline', message: 'No internet connection' }),
        {
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
    
    return new Response('Offline', { status: 503 });
  }
}

// ──────────────────────────────────────────────────────────────────────────
// BACKGROUND SYNC - Sincronizar respuestas offline
// ──────────────────────────────────────────────────────────────────────────
self.addEventListener('sync', event => {
  console.log('[SW] Background Sync:', event.tag);
  
  if (event.tag === 'sync-interview-answers') {
    event.waitUntil(syncInterviewAnswers());
  }
  
  if (event.tag === 'sync-user-data') {
    event.waitUntil(syncUserData());
  }
});

/**
 * Sincroniza respuestas de entrevista guardadas offline
 */
async function syncInterviewAnswers() {
  try {
    console.log('[SW] Syncing interview answers...');
    
    // Abrir IndexedDB con respuestas pendientes
    const db = await openIndexedDB('ready4hire-offline');
    const answers = await getAllFromStore(db, 'pending-answers');
    
    console.log(`[SW] Found ${answers.length} pending answers`);
    
    let synced = 0;
    let failed = 0;
    
    for (const answer of answers) {
      try {
        const response = await fetch('/api/v2/interviews/sync', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(answer)
        });
        
        if (response.ok) {
          await deleteFromStore(db, 'pending-answers', answer.id);
          synced++;
          console.log('[SW] Synced answer:', answer.id);
        } else {
          failed++;
        }
      } catch (error) {
        console.error('[SW] Failed to sync answer:', error);
        failed++;
      }
    }
    
    console.log(`[SW] Sync complete: ${synced} synced, ${failed} failed`);
    
    // Notificar al usuario
    if (synced > 0) {
      self.registration.showNotification('Ready4Hire', {
        body: `✅ ${synced} respuestas sincronizadas`,
        icon: '/images/icons/icon-192x192.png',
        badge: '/images/icons/badge-72x72.png'
      });
    }
    
  } catch (error) {
    console.error('[SW] Sync error:', error);
  }
}

/**
 * Sincroniza datos de usuario
 */
async function syncUserData() {
  console.log('[SW] Syncing user data...');
  // Implementar lógica de sync de skills, perfil, etc.
}

// ──────────────────────────────────────────────────────────────────────────
// PUSH NOTIFICATIONS
// ──────────────────────────────────────────────────────────────────────────
self.addEventListener('push', event => {
  console.log('[SW] Push notification received');
  
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'Ready4Hire';
  const options = {
    body: data.body || 'Nueva notificación',
    icon: '/images/icons/icon-192x192.png',
    badge: '/images/icons/badge-72x72.png',
    vibrate: [200, 100, 200],
    data: data,
    actions: [
      { action: 'open', title: 'Abrir' },
      { action: 'close', title: 'Cerrar' }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

self.addEventListener('notificationclick', event => {
  console.log('[SW] Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      clients.openWindow(event.notification.data.url || '/')
    );
  }
});

// ──────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS - IndexedDB
// ──────────────────────────────────────────────────────────────────────────

/**
 * Abre conexión con IndexedDB
 */
function openIndexedDB(dbName) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = event => {
      const db = event.target.result;
      
      if (!db.objectStoreNames.contains('pending-answers')) {
        db.createObjectStore('pending-answers', { keyPath: 'id', autoIncrement: true });
      }
      
      if (!db.objectStoreNames.contains('cached-questions')) {
        db.createObjectStore('cached-questions', { keyPath: 'id' });
      }
    };
  });
}

/**
 * Obtiene todos los registros de un store
 */
function getAllFromStore(db, storeName) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);
    const request = store.getAll();
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Elimina un registro de un store
 */
function deleteFromStore(db, storeName, id) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.delete(id);
    
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// ──────────────────────────────────────────────────────────────────────────
// MESSAGE HANDLER - Comunicación con la app
// ──────────────────────────────────────────────────────────────────────────
self.addEventListener('message', event => {
  console.log('[SW] Message received:', event.data);
  
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(RUNTIME_CACHE)
        .then(cache => cache.addAll(event.data.urls))
    );
  }
  
  if (event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys()
        .then(names => Promise.all(names.map(name => caches.delete(name))))
    );
  }
});

console.log('[SW] Service Worker loaded successfully');

