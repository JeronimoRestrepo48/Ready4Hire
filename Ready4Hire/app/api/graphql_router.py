"""
GraphQL Router
Configuración de FastAPI con Strawberry GraphQL
"""

from fastapi import APIRouter
from strawberry.fastapi import GraphQLRouter
from app.api.graphql_schema import schema

# Crear router de FastAPI para GraphQL
router = APIRouter()

# Configurar GraphQL endpoint
graphql_app = GraphQLRouter(
    schema=schema,
    path="/graphql",
    graphql_ide="graphiql",  # Enable GraphiQL interface for development (updated from graphiql parameter)
)

# Include GraphQL router
router.include_router(graphql_app, prefix="", tags=["GraphQL"])
