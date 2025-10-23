"""
Sentry Integration for Error Tracking and Monitoring.
Integración completa con Sentry para monitoreo en producción.
"""
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def init_sentry(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
    enable_tracing: bool = True
) -> None:
    """
    Inicializa Sentry para monitoreo de errores.
    
    Args:
        dsn: Sentry DSN (default: desde settings)
        environment: Entorno (development, staging, production)
        traces_sample_rate: % de transacciones a muestrear (0.0-1.0)
        profiles_sample_rate: % de perfiles a capturar (0.0-1.0)
        enable_tracing: Si habilitar performance tracing
    
    Example:
        # En producción
        init_sentry(
            dsn="https://abc123@o123.ingest.sentry.io/456",
            environment="production",
            traces_sample_rate=1.0  # 100% en producción
        )
    """
    # Solo inicializar si tenemos DSN
    sentry_dsn = dsn or getattr(settings, 'SENTRY_DSN', None)
    
    if not sentry_dsn:
        logger.warning("⚠️ Sentry DSN not configured. Error tracking disabled.")
        return
    
    # Determinar environment
    env = environment or settings.ENVIRONMENT
    
    # Configurar integraciones
    integrations = [
        # FastAPI integration (captura errores HTTP)
        FastApiIntegration(
            transaction_style="endpoint",  # Agrupar por endpoint
            failed_request_status_codes=[500, 599]  # Solo errores 5xx
        ),
        
        # Logging integration (captura logs de ERROR y CRITICAL)
        LoggingIntegration(
            level=logging.INFO,  # Capturar desde INFO
            event_level=logging.ERROR  # Crear eventos desde ERROR
        )
    ]
    
    # Inicializar Sentry
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=env,
        integrations=integrations,
        
        # Performance Monitoring
        traces_sample_rate=traces_sample_rate if enable_tracing else 0.0,
        profiles_sample_rate=profiles_sample_rate if enable_tracing else 0.0,
        
        # Release tracking
        release=f"ready4hire@{settings.APP_VERSION}",
        
        # Opciones adicionales
        send_default_pii=False,  # No enviar PII por defecto
        attach_stacktrace=True,  # Incluir stack traces
        max_breadcrumbs=50,  # Historial de eventos
        
        # Filtros
        before_send=before_send_filter,
        before_breadcrumb=before_breadcrumb_filter,
    )
    
    logger.info(
        f"✅ Sentry initialized: env={env}, "
        f"traces={traces_sample_rate}, profiles={profiles_sample_rate}"
    )


def before_send_filter(event, hint):
    """
    Filtro para eventos antes de enviar a Sentry.
    Permite filtrar errores que no queremos trackear.
    
    Args:
        event: Evento de Sentry
        hint: Información adicional
        
    Returns:
        event si debe enviarse, None si debe descartarse
    """
    # Ignorar ciertos tipos de errores
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        
        # Ignorar errores de validación de Pydantic (son errores de usuario)
        if exc_type.__name__ == 'ValidationError':
            return None
        
        # Ignorar errores de timeout (son esperados)
        if 'timeout' in str(exc_value).lower():
            return None
        
        # Ignorar errores de circuit breaker OPEN (son esperados)
        if 'circuit' in str(exc_value).lower() and 'open' in str(exc_value).lower():
            return None
    
    # Enviar el resto
    return event


def before_breadcrumb_filter(crumb, hint):
    """
    Filtro para breadcrumbs antes de agregar.
    Permite filtrar ruido en los breadcrumbs.
    
    Args:
        crumb: Breadcrumb
        hint: Información adicional
        
    Returns:
        crumb si debe agregarse, None si debe descartarse
    """
    # Ignorar queries de health check
    if crumb.get('category') == 'httplib':
        if '/health' in crumb.get('data', {}).get('url', ''):
            return None
    
    return crumb


def capture_exception(
    error: Exception,
    level: str = "error",
    extra: Optional[dict] = None
) -> Optional[str]:
    """
    Captura una excepción manualmente en Sentry.
    
    Args:
        error: Excepción a capturar
        level: Nivel de severidad (error, warning, info)
        extra: Datos adicionales para contexto
        
    Returns:
        Event ID de Sentry o None
    
    Example:
        try:
            risky_operation()
        except Exception as e:
            capture_exception(e, extra={"user_id": "123", "action": "risky"})
    """
    with sentry_sdk.push_scope() as scope:
        # Agregar contexto adicional
        if extra:
            for key, value in extra.items():
                scope.set_extra(key, value)
        
        # Configurar nivel
        scope.level = level
        
        # Capturar excepción
        event_id = sentry_sdk.capture_exception(error)
        
        if event_id:
            logger.debug(f"Exception captured in Sentry: {event_id}")
        
        return event_id


def capture_message(
    message: str,
    level: str = "info",
    extra: Optional[dict] = None
) -> Optional[str]:
    """
    Captura un mensaje manualmente en Sentry.
    
    Args:
        message: Mensaje a capturar
        level: Nivel de severidad (error, warning, info)
        extra: Datos adicionales para contexto
        
    Returns:
        Event ID de Sentry o None
    
    Example:
        capture_message(
            "User completed interview",
            level="info",
            extra={"user_id": "123", "score": 8.5}
        )
    """
    with sentry_sdk.push_scope() as scope:
        # Agregar contexto adicional
        if extra:
            for key, value in extra.items():
                scope.set_extra(key, value)
        
        # Configurar nivel
        scope.level = level
        
        # Capturar mensaje
        event_id = sentry_sdk.capture_message(message, level)
        
        return event_id


def set_user_context(user_id: str, username: Optional[str] = None, email: Optional[str] = None):
    """
    Establece contexto de usuario para Sentry.
    
    Args:
        user_id: ID del usuario
        username: Nombre de usuario (opcional)
        email: Email del usuario (opcional)
    
    Example:
        set_user_context(user_id="user-123", email="user@example.com")
    """
    sentry_sdk.set_user({
        "id": user_id,
        "username": username,
        "email": email
    })


def set_context(key: str, data: dict):
    """
    Establece contexto adicional para Sentry.
    
    Args:
        key: Nombre del contexto (ej: "interview", "llm")
        data: Datos del contexto
    
    Example:
        set_context("interview", {
            "interview_id": "int-123",
            "role": "Backend Developer",
            "phase": "technical"
        })
    """
    sentry_sdk.set_context(key, data)


def add_breadcrumb(message: str, category: str = "default", level: str = "info", data: Optional[dict] = None):
    """
    Agrega un breadcrumb manualmente.
    
    Args:
        message: Mensaje del breadcrumb
        category: Categoría (ej: "auth", "llm", "database")
        level: Nivel (info, warning, error)
        data: Datos adicionales
    
    Example:
        add_breadcrumb(
            message="Question selected",
            category="interview",
            data={"question_id": "q-42", "difficulty": "mid"}
        )
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


# Decorador para funciones críticas
def monitor_function(func):
    """
    Decorador para monitorear funciones con Sentry.
    Captura automáticamente excepciones y performance.
    
    Example:
        @monitor_function
        async def critical_operation():
            # código crítico
            pass
    """
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        with sentry_sdk.start_transaction(op="function", name=func.__name__):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                capture_exception(e, extra={
                    "function": func.__name__,
                    "args": str(args)[:200],  # Limitar tamaño
                    "kwargs": str(kwargs)[:200]
                })
                raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        with sentry_sdk.start_transaction(op="function", name=func.__name__):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                capture_exception(e, extra={
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                })
                raise
    
    # Determinar si es async o sync
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

