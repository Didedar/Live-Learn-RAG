-- Migration 002: Feedback System Tables
-- This migration adds tables for the feedback learning system

-- Message sessions for tracking question-answer pairs
CREATE TABLE IF NOT EXISTS message_sessions (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(128),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    contexts_used JSON DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_message_sessions_session_id ON message_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_message_sessions_created_at ON message_sessions(created_at);

-- Feedback events for user corrections
CREATE TABLE IF NOT EXISTS feedback_events (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36) NOT NULL,
    label VARCHAR(20) NOT NULL CHECK (label IN ('correct', 'partially_correct', 'incorrect')),
    correction_text TEXT,
    scope VARCHAR(10) NOT NULL CHECK (scope IN ('chunk', 'doc', 'global')),
    reason TEXT,
    target_doc_id INTEGER,
    target_chunk_id INTEGER,
    user_id VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES message_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_events_message_id ON feedback_events(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_events_label ON feedback_events(label);
CREATE INDEX IF NOT EXISTS idx_feedback_events_created_at ON feedback_events(created_at);

-- Index mutations for tracking knowledge base changes
CREATE TABLE IF NOT EXISTS index_mutations (
    id VARCHAR(36) PRIMARY KEY,
    feedback_event_id VARCHAR(36) NOT NULL,
    operation VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'applied', 'failed', 'reverted')),
    affected_doc_id INTEGER,
    affected_chunk_id INTEGER,
    operation_data JSON DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP,
    FOREIGN KEY (feedback_event_id) REFERENCES feedback_events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_index_mutations_feedback_event_id ON index_mutations(feedback_event_id);
CREATE INDEX IF NOT EXISTS idx_index_mutations_status ON index_mutations(status);
CREATE INDEX IF NOT EXISTS idx_index_mutations_operation ON index_mutations(operation);

-- Chunk weights for feedback-based ranking
CREATE TABLE IF NOT EXISTS chunk_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER UNIQUE NOT NULL,
    penalty_weight REAL DEFAULT 0.0,
    boost_weight REAL DEFAULT 0.0,
    is_deprecated BOOLEAN DEFAULT FALSE,
    feedback_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunk_weights_chunk_id ON chunk_weights(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_weights_is_deprecated ON chunk_weights(is_deprecated);
CREATE INDEX IF NOT EXISTS idx_chunk_weights_last_updated ON chunk_weights(last_updated);

-- Add source and version columns to chunks table (if not exists)
ALTER TABLE chunks ADD COLUMN source VARCHAR(50) DEFAULT 'original';
ALTER TABLE chunks ADD COLUMN version INTEGER DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id_ordinal ON chunks(document_id, ordinal);

-- Performance indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_not_null ON chunks(id) WHERE embedding IS NOT NULL AND embedding != '[]';

-- Add migration record
INSERT OR IGNORE INTO migrations (name) VALUES ('002_feedback_tables');