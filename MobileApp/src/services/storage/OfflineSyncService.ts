/**
 * Offline Sync Service
 * Maneja sincronización de datos offline/online
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { apolloClient } from '../api/apolloClient';

interface QueuedOperation {
  id: string;
  type: 'mutation' | 'query';
  operationName: string;
  variables: any;
  timestamp: number;
  retries: number;
}

const QUEUE_KEY = '@sync_queue';
const CACHE_PREFIX = '@cache_';
const MAX_RETRIES = 3;

class OfflineSyncService {
  private syncQueue: QueuedOperation[] = [];
  private isSyncing: boolean = false;
  private listeners: Array<(isOnline: boolean) => void> = [];

  constructor() {
    this.init();
  }

  /**
   * Inicializar el servicio
   */
  async init() {
    // Cargar queue desde storage
    await this.loadQueue();

    // Escuchar cambios de conectividad
    NetInfo.addEventListener(state => {
      const isOnline = state.isConnected ?? false;
      
      // Notificar a listeners
      this.listeners.forEach(listener => listener(isOnline));

      // Sincronizar si está online
      if (isOnline && this.syncQueue.length > 0) {
        this.syncPendingOperations();
      }
    });
  }

  /**
   * Agregar listener para cambios de conectividad
   */
  addConnectivityListener(listener: (isOnline: boolean) => void) {
    this.listeners.push(listener);
  }

  /**
   * Remover listener
   */
  removeConnectivityListener(listener: (isOnline: boolean) => void) {
    this.listeners = this.listeners.filter(l => l !== listener);
  }

  /**
   * Verificar si hay conexión
   */
  async isOnline(): Promise<boolean> {
    const state = await NetInfo.fetch();
    return state.isConnected ?? false;
  }

  /**
   * Guardar dato en caché
   */
  async cacheData(key: string, data: any): Promise<void> {
    try {
      const cacheKey = `${CACHE_PREFIX}${key}`;
      await AsyncStorage.setItem(cacheKey, JSON.stringify({
        data,
        timestamp: Date.now(),
      }));
    } catch (error) {
      console.error('Error caching data:', error);
    }
  }

  /**
   * Obtener dato de caché
   */
  async getCachedData(key: string, maxAge: number = 3600000): Promise<any | null> {
    try {
      const cacheKey = `${CACHE_PREFIX}${key}`;
      const cached = await AsyncStorage.getItem(cacheKey);
      
      if (!cached) return null;

      const { data, timestamp } = JSON.parse(cached);
      const age = Date.now() - timestamp;

      // Si el dato es muy viejo, retornar null
      if (age > maxAge) {
        await AsyncStorage.removeItem(cacheKey);
        return null;
      }

      return data;
    } catch (error) {
      console.error('Error getting cached data:', error);
      return null;
    }
  }

  /**
   * Limpiar caché
   */
  async clearCache(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter(key => key.startsWith(CACHE_PREFIX));
      await AsyncStorage.multiRemove(cacheKeys);
    } catch (error) {
      console.error('Error clearing cache:', error);
    }
  }

  /**
   * Agregar operación a la queue
   */
  async queueOperation(
    type: 'mutation' | 'query',
    operationName: string,
    variables: any
  ): Promise<string> {
    const operation: QueuedOperation = {
      id: `${Date.now()}_${Math.random()}`,
      type,
      operationName,
      variables,
      timestamp: Date.now(),
      retries: 0,
    };

    this.syncQueue.push(operation);
    await this.saveQueue();

    // Intentar sincronizar inmediatamente si hay conexión
    if (await this.isOnline()) {
      this.syncPendingOperations();
    }

    return operation.id;
  }

  /**
   * Cargar queue desde storage
   */
  private async loadQueue(): Promise<void> {
    try {
      const queueJson = await AsyncStorage.getItem(QUEUE_KEY);
      if (queueJson) {
        this.syncQueue = JSON.parse(queueJson);
      }
    } catch (error) {
      console.error('Error loading sync queue:', error);
    }
  }

  /**
   * Guardar queue en storage
   */
  private async saveQueue(): Promise<void> {
    try {
      await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(this.syncQueue));
    } catch (error) {
      console.error('Error saving sync queue:', error);
    }
  }

  /**
   * Sincronizar operaciones pendientes
   */
  async syncPendingOperations(): Promise<void> {
    if (this.isSyncing || this.syncQueue.length === 0) {
      return;
    }

    this.isSyncing = true;

    try {
      const operations = [...this.syncQueue];

      for (const operation of operations) {
        try {
          if (operation.type === 'mutation') {
            // Ejecutar mutation
            // TODO: Implementar según la mutation específica
            console.log(`Syncing mutation: ${operation.operationName}`);
          } else {
            // Ejecutar query
            console.log(`Syncing query: ${operation.operationName}`);
          }

          // Remover de queue si fue exitoso
          this.syncQueue = this.syncQueue.filter(op => op.id !== operation.id);
        } catch (error) {
          console.error(`Error syncing operation ${operation.id}:`, error);

          // Incrementar retries
          operation.retries++;

          // Si excedió max retries, remover
          if (operation.retries >= MAX_RETRIES) {
            console.warn(`Operation ${operation.id} exceeded max retries, removing`);
            this.syncQueue = this.syncQueue.filter(op => op.id !== operation.id);
          }
        }
      }

      await this.saveQueue();
    } finally {
      this.isSyncing = false;
    }
  }

  /**
   * Obtener número de operaciones pendientes
   */
  getPendingCount(): number {
    return this.syncQueue.length;
  }

  /**
   * Limpiar queue
   */
  async clearQueue(): Promise<void> {
    this.syncQueue = [];
    await AsyncStorage.removeItem(QUEUE_KEY);
  }
}

// Exportar instancia única
export const offlineSyncService = new OfflineSyncService();
export default offlineSyncService;

