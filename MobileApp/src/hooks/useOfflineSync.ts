/**
 * useOfflineSync Hook
 * Custom hook para usar el servicio de sincronización offline
 */

import { useState, useEffect, useCallback } from 'react';
import { offlineSyncService } from '../services/storage/OfflineSyncService';

export const useOfflineSync = () => {
  const [isOnline, setIsOnline] = useState(true);
  const [pendingCount, setPendingCount] = useState(0);
  const [isSyncing, setIsSyncing] = useState(false);

  useEffect(() => {
    // Listener para cambios de conectividad
    const handleConnectivityChange = (online: boolean) => {
      setIsOnline(online);
    };

    offlineSyncService.addConnectivityListener(handleConnectivityChange);

    // Verificar estado inicial
    offlineSyncService.isOnline().then(setIsOnline);

    // Actualizar conteo pendiente
    const updatePendingCount = () => {
      setPendingCount(offlineSyncService.getPendingCount());
    };

    updatePendingCount();
    const interval = setInterval(updatePendingCount, 1000);

    return () => {
      offlineSyncService.removeConnectivityListener(handleConnectivityChange);
      clearInterval(interval);
    };
  }, []);

  /**
   * Ejecutar operación con soporte offline
   */
  const executeWithOfflineSupport = useCallback(
    async <T,>(
      operation: () => Promise<T>,
      fallback?: () => Promise<T | null>
    ): Promise<T | null> => {
      try {
        // Intentar ejecutar la operación
        return await operation();
      } catch (error) {
        console.error('Operation failed:', error);

        // Si hay fallback, ejecutarlo
        if (fallback) {
          return await fallback();
        }

        return null;
      }
    },
    []
  );

  /**
   * Cachear dato
   */
  const cacheData = useCallback(async (key: string, data: any) => {
    await offlineSyncService.cacheData(key, data);
  }, []);

  /**
   * Obtener dato cacheado
   */
  const getCachedData = useCallback(
    async (key: string, maxAge?: number) => {
      return await offlineSyncService.getCachedData(key, maxAge);
    },
    []
  );

  /**
   * Encolar operación
   */
  const queueOperation = useCallback(
    async (type: 'mutation' | 'query', operationName: string, variables: any) => {
      return await offlineSyncService.queueOperation(type, operationName, variables);
    },
    []
  );

  /**
   * Forzar sincronización
   */
  const forceSync = useCallback(async () => {
    setIsSyncing(true);
    try {
      await offlineSyncService.syncPendingOperations();
    } finally {
      setIsSyncing(false);
    }
  }, []);

  /**
   * Limpiar caché
   */
  const clearCache = useCallback(async () => {
    await offlineSyncService.clearCache();
  }, []);

  return {
    isOnline,
    pendingCount,
    isSyncing,
    executeWithOfflineSupport,
    cacheData,
    getCachedData,
    queueOperation,
    forceSync,
    clearCache,
  };
};

