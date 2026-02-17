-- ============================================================================
-- AtlasRAG PostgreSQL Initialization Script
-- ============================================================================
-- This script runs once when the Postgres container is first created.
-- It sets up the database schema for metadata storage.
--
-- Extensions:
-- - uuid-ossp: For UUID generation
-- - pg_trgm: For full-text search on text fields
--
-- Schemas:
-- - atlasrag: Main application schema
--
-- TODO: Phase 5 - Document metadata tables
--   - documents: Track ingested documents
--   - chunks: Store document chunks with embeddings
--   - ingestion_jobs: Track processing status
--
-- TODO: Phase 9 - API and User Management
--   - api_keys: API key management
--   - user_sessions: User sessions and auth
--   - request_logs: API request tracking
--
-- TODO: Phase 11 - Evaluation Tracking
--   - evaluation_results: Store metrics over time
--   - baseline_metrics: Track performance baselines
-- ============================================================================

-- Set timezone to UTC
SET TIME ZONE 'UTC';

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For full-text search

-- ============================================================================
-- Application Schema
-- ============================================================================

-- Create application schema
CREATE SCHEMA IF NOT EXISTS atlasrag;

-- Set schema search path
SET search_path TO atlasrag, public;

-- ============================================================================
-- TODO: Phase 5 - Document Metadata Tables
-- ============================================================================
-- These tables will store metadata about documents in the system.
--
-- Planned tables:
-- - documents: Document metadata (filename, type, size, upload time, etc.)
-- - chunks: Document chunks with embeddings
-- - ingestion_jobs: Track ingestion job status and progress
--
-- Example structure (to be implemented in Phase 5):
--
-- CREATE TABLE documents (
--     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     filename VARCHAR(255) NOT NULL,
--     file_type VARCHAR(50) NOT NULL,  -- pdf, docx, txt, html, md
--     file_size_bytes BIGINT NOT NULL,
--     content_hash VARCHAR(64) UNIQUE,  -- SHA256 hash to detect duplicates
--     upload_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
--     processing_status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
--     metadata JSONB,  -- Custom metadata (author, source, etc.)
--     created_at TIMESTAMP NOT NULL DEFAULT NOW(),
--     updated_at TIMESTAMP NOT NULL DEFAULT NOW()
-- );
--
-- CREATE TABLE chunks (
--     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
--     chunk_number INTEGER NOT NULL,  -- Position in document
--     content TEXT NOT NULL,
--     embedding FLOAT8[] NOT NULL,  -- Vector embedding for semantic search
--     metadata JSONB,  -- Page number, section, etc.
--     created_at TIMESTAMP NOT NULL DEFAULT NOW()
-- );

-- ============================================================================
-- TODO: Phase 9 - API Authentication Tables
-- ============================================================================
-- These tables will manage API access and user sessions.
--
-- Planned tables:
-- - api_keys: API key management
-- - user_sessions: User login sessions
-- - audit_logs: Track API calls for compliance

-- ============================================================================
-- TODO: Phase 11 - Evaluation & Metrics
-- ============================================================================
-- These tables will store evaluation results and metrics over time.
--
-- Planned tables:
-- - evaluation_results: RAGAS metrics (faithfulness, recall, etc.)
-- - metric_baselines: Track performance expectations
-- - query_logs: Store queries and responses for analysis

-- ============================================================================
-- Grants and Permissions
-- ============================================================================

-- Grant schema permissions to application user
GRANT USAGE ON SCHEMA atlasrag TO atlasrag;

-- Grant table permissions (once tables are created)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA atlasrag TO atlasrag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA atlasrag TO atlasrag;

-- Future tables created in this schema should also be accessible
ALTER DEFAULT PRIVILEGES IN SCHEMA atlasrag
    GRANT ALL PRIVILEGES ON TABLES TO atlasrag;

ALTER DEFAULT PRIVILEGES IN SCHEMA atlasrag
    GRANT ALL PRIVILEGES ON SEQUENCES TO atlasrag;

-- ============================================================================
-- Initialization Complete
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'AtlasRAG database initialized successfully!';
    RAISE NOTICE 'Schema: atlasrag';
    RAISE NOTICE 'User: atlasrag';
    RAISE NOTICE 'Database: atlasrag';
END $$;
