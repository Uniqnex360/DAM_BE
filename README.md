# DAM Backend

A FastAPI-based backend for a Digital Asset Management (DAM) system. This project provides authenticated asset upload, image processing, analytics, room visualization, and database-backed user/project management.

## Overview

This repository contains the backend service for a DAM application. It exposes REST APIs for user authentication, asset uploads, processing operations, analytics, and room visualization workflows.

Key capabilities:
- JWT authentication with login/register endpoints
- Asset upload with Cloudinary integration
- Image analysis and processing workflows
- Project and user-based asset organization
- PostgreSQL database storage via SQLAlchemy async sessions
- Docker Compose support for local development

## Architecture

The project is organized into the following main areas:

- `app/main.py`: FastAPI application entrypoint, middleware, static file mounting, and router registration.
- `app/api/v1/router.py`: API router that includes endpoint groups.
- `app/api/v1/endpoints/`: REST endpoint modules for auth, assets, reports, users, dashboard analytics, and room visualizer.
- `app/core/`: configuration and security helpers.
- `app/db/`: database session creation and access.
- `app/models/`: ORM models for users, assets, uploads, projects, processing results, and static asset entities.
- `app/schemas/`: Pydantic request and response schemas.
- `app/services/`: business logic for image processing, media upload, quality analysis, statistics, and repositories.
- `app/static/`: local static files for uploads and processed assets.
- `alembic/`: database migration configuration and scripts.

## Repository Structure

Root files:
- `Dockerfile`: backend container definition.
- `docker-compose.yml`: development stack with backend, PostgreSQL, and Redis.
- `requirements.txt`: Python dependencies.
- `package.json`: frontend 3D/web package dependencies (for integration or UI projects).
- `alembic.ini`: migration configuration.

Important directories:
- `app/`: backend source code.
- `app/api/v1/endpoints/`: API route implementations.
- `app/core/`: settings and security utilities.
- `app/db/`: async SQLAlchemy session factory.
- `app/models/`: SQLAlchemy ORM models.
- `app/services/`: processing and business services.
- `static/`: runtime-generated uploads and processed assets.
- `app/static/`: application static mount path.

## Setup

### Environment

Create a `.env` file at the repository root or export environment variables manually.

Required configuration values:
- `PROJECT_NAME`: application name
- `API_V1_STR`: API prefix, e.g. `/api/v1`
- `POSTGRES_SERVER`: database host
- `POSTGRES_USER`: database username
- `POSTGRES_PASSWORD`: database password
- `POSTGRES_DB`: database name
- `DATABASE_URL`: SQLAlchemy DB URL, or built from the POSTGRES_ values
- `SECRET_KEY`: JWT signing secret
- `ALGORITHM`: JWT algorithm, e.g. `HS256`
- `ACCESS_TOKEN_EXPIRE_MINUTES`: token lifetime in minutes
- `STORAGE_PROVIDER`: storage backend, e.g. `local`

Optional values:
- `HF_TOKEN`: Hugging Face token for model downloads
- `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`, `CLOUDINARY_UPLOAD_PRESET`
- `REDIS_HOST`: Redis host for caching or background tasks

Example `.env` content:

```env
PROJECT_NAME=Digital-Assets-Management
API_V1_STR=/api/v1
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=dam_db
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/dam_db
SECRET_KEY=unsafe_default_key_change_in_env
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
STORAGE_PROVIDER=local
HF_TOKEN=
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
CLOUDINARY_UPLOAD_PRESET=
REDIS_HOST=localhost
```

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run with Docker Compose

```bash
docker compose up --build
```

The backend will be available at `http://127.0.0.1:8002` by default.

## Running Locally Without Docker

1. Ensure PostgreSQL is running and reachable.
2. Set your `.env` values.
3. Start the application:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

The application exposes the following main endpoint groups under the configured API prefix (`/api/v1`):

- `/auth`: user authentication and token management
  - `POST /login`: email/password login
  - `POST /register`: user signup
  - `GET /verify`: validate JWT and retrieve current user info
  - `POST /impersonate/{user_id}`: admin impersonation
  - `POST /stop-impersonation`: stop impersonation

- `/assets`: asset upload, processing, and analysis
  - `POST /upload`: upload images and files
  - `POST /analyze`: request image quality analysis
  - `POST /{image_id}/process`: process an uploaded image

- `/users`: user management routes
- `/reports`: reporting and analytics
- `/dashboard`: analytics/dashboard data
- `/room-visualizer`: room visualization endpoints

> Use `/docs` for interactive Swagger UI documentation once the service is running.

## Database

The backend uses PostgreSQL and Alembic for migrations.

- `app/db/session.py`: async SQLAlchemy engine and session factory.
- `app/models/`: ORM models including `User`, `Upload`, `Image`, `Project`, `Model3D`, `ARAsset`, and `Texture`.

Run migrations with Alembic:

```bash
alembic upgrade head
```

## Image Processing and Services

Core image workflow responsibilities are in `app/services/`: 

- `image_processor.py`: image processing, OCR, background removal, watermark detection, text removal, and inpainting.
- `media.py`: cloud upload and storage helpers.
- `quality_analyzer.py`: image quality analysis logic.
- `statistics.py`: processing stats and usage tracking.
- `repositories.py`: database repository utilities.

## Static Files and Uploads

The application mounts static files under `/static` and persistently stores uploads in:

- `static/uploads`
- `static/processed`

The app will create these directories automatically when started if they are missing.

## Notes

- `app/main.py` mounts the static directory twice for compatibility and serves the FastAPI app at root.
- Environment variables are loaded using `pydantic-settings` from a `.env` file.
- The project includes Hugging Face and Cloudinary support for advanced image tasks.

## Contribution

If you contribute or extend the backend, keep the following in mind:
- Use async SQLAlchemy sessions in route dependencies.
- Keep image processing work off the main thread where possible.
- Preserve the `app/api/v1` router structure for versioned APIs.
- Document new endpoints in the OpenAPI docs via Pydantic response/request models.
