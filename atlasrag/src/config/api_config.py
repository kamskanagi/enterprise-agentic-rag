"""API Server Configuration Models

Frozen BaseModel classes for API server, database, and security configurations.
"""

from pydantic import BaseModel, SecretStr


class CORSConfig(BaseModel, frozen=True):
    """CORS (Cross-Origin Resource Sharing) configuration."""

    origins: list[str]


class DatabaseConfig(BaseModel, frozen=True):
    """PostgreSQL database connection and pool configuration."""

    url: SecretStr
    pool_size: int
    max_overflow: int


class APIConfig(BaseModel, frozen=True):
    """Unified API server configuration."""

    host: str
    port: int
    workers: int
    reload: bool
    debug: bool
    cors: CORSConfig
    database: DatabaseConfig
    api_key: SecretStr
    jwt_secret: SecretStr
    jwt_algorithm: str
    jwt_expiration_minutes: int
