# Alembic Database Migrations

This directory contains Alembic configuration for database migrations in the Python backend.

## Setup

1. Install dependencies:
```bash
pip install sqlalchemy alembic
```

2. Initialize (already done):
```bash
alembic init alembic
```

3. Create your first migration:
```bash
cd Ready4Hire
alembic revision --autogenerate -m "Initial schema"
```

4. Apply migrations:
```bash
alembic upgrade head
```

5. Rollback:
```bash
alembic downgrade -1
```

## Configuration

- **alembic.ini**: Main configuration file
- **alembic/env.py**: Environment setup and connection
- **alembic/versions/**: Migration files directory

## Important Notes

- The backend currently uses **memory-based repositories** for interviews (stateless)
- Migrations are for future PostgreSQL integration if needed
- Base models are defined in `app/infrastructure/persistence/base.py`
- Connection string is read from `app/config.py` settings

## Creating Migrations

```bash
# Auto-generate from models
alembic revision --autogenerate -m "Description"

# Manual migration
alembic revision -m "Description"
```

## Database Models

Currently, the domain uses entities that don't map directly to SQLAlchemy models. To use migrations:

1. Create SQLAlchemy models in `app/infrastructure/persistence/models.py`
2. Import them in `alembic/env.py`
3. Generate migrations with `alembic revision --autogenerate`

