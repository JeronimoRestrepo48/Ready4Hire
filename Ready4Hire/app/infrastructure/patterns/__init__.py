"""
Patrones de Diseño para Ready4Hire Infrastructure.

Este módulo proporciona implementaciones de patrones de diseño comunes
para mejorar la modularidad, testabilidad y mantenibilidad del código.
"""

from app.infrastructure.patterns.factory import (
    LLMServiceFactory,
    CacheServiceFactory,
    MLServiceFactory,
    RepositoryFactory,
)
from app.infrastructure.patterns.strategy import (
    EvaluationStrategy,
    FeedbackStrategy,
    QuestionSelectionStrategy,
    StrategyRegistry,
)
from app.infrastructure.patterns.repository import (
    BaseRepository,
    RepositoryRegistry,
)
from app.infrastructure.patterns.adapter import (
    ServiceAdapter,
    LLMAdapter,
    CacheAdapter,
)
from app.infrastructure.patterns.observer import (
    EventObserver,
    EventPublisher,
    EventBus,
)
from app.infrastructure.patterns.builder import (
    InterviewBuilder,
    EvaluationBuilder,
    QuestionSelectorBuilder,
)
from app.infrastructure.patterns.decorator import (
    CachedService,
    LoggedService,
    RetryService,
    MetricsService,
    log_method_call,
    retry_on_failure,
)
from app.infrastructure.patterns.facade import (
    LLMFacade,
    MLFacade,
    InfrastructureFacade,
)
from app.infrastructure.patterns.proxy import (
    LazyServiceProxy,
    CachedServiceProxy,
)

__all__ = [
    # Factories
    "LLMServiceFactory",
    "CacheServiceFactory",
    "MLServiceFactory",
    "RepositoryFactory",
    # Strategies
    "EvaluationStrategy",
    "FeedbackStrategy",
    "QuestionSelectionStrategy",
    "StrategyRegistry",
    # Repository
    "BaseRepository",
    "RepositoryRegistry",
    # Adapters
    "ServiceAdapter",
    "LLMAdapter",
    "CacheAdapter",
    # Observer
    "EventObserver",
    "EventPublisher",
    "EventBus",
    # Builders
    "InterviewBuilder",
    "EvaluationBuilder",
    "QuestionSelectorBuilder",
    # Decorators
    "CachedService",
    "LoggedService",
    "RetryService",
    "MetricsService",
    # Facades
    "LLMFacade",
    "MLFacade",
    "InfrastructureFacade",
    # Proxies
    "LazyServiceProxy",
    "CachedServiceProxy",
]

