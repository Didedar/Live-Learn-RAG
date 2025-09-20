-- Migration: Separated feedback system with intent-based targeting
-- Creates tables for intent keys, separated feedback storage, and application tracking

-- Intent keys table for normalized intent storage
CREATE TABLE IF NOT EXISTS intent_keys (
    id VARCHAR(64) PRIMARY KEY,  -- SHA-256 hash of normalized intent
    normalized_text TEXT NOT NULL,
    entities JSON DEFAULT '[]',  -- Extracted entities
    tokens JSON DEFAULT '[]',    -- Normalized tokens
    embedding JSON NOT NULL,     -- Intent embedding
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_intent_keys_normalized (normalized_text),
    INDEX idx_intent_keys_created (created_at)
);

-- Separated feedback storage (isolated from documents)
CREATE TABLE IF NOT EXISTS intent_feedback (
    id VARCHAR(36) PRIMARY KEY,
    
    -- Intent binding
    intent_key VARCHAR(64) NOT NULL,
    query_text TEXT NOT NULL,
    
    -- User and session info
    user_id VARCHAR(128),
    session_id VARCHAR(128),
    message_id VARCHAR(36) NOT NULL,
    
    -- Feedback details
    label ENUM('prefer', 'reject', 'fix', 'style') NOT NULL,
    polarity INTEGER NOT NULL,  -- +1 for positive, -1 for negative
    weight FLOAT DEFAULT 1.0,   -- Trust/confidence weight
    scope ENUM('local', 'cluster', 'global') DEFAULT 'local',
    
    -- Evidence linking (what docs/chunks this applies to)
    evidence JSON DEFAULT '[]',  -- [{"doc_id": 45, "chunk_id": 3, "offsets": [...]}]
    
    -- Feedback content
    notes TEXT,
    correction_text TEXT,
    
    -- Quality control
    is_verified BOOLEAN DEFAULT FALSE,
    verification_count INTEGER DEFAULT 0,
    is_spam BOOLEAN DEFAULT FALSE,
    confidence_score FLOAT DEFAULT 1.0,
    
    -- TTL and decay
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    decay_factor FLOAT DEFAULT 1.0,
    
    -- Foreign key constraint
    FOREIGN KEY (intent_key) REFERENCES intent_keys(id) ON DELETE CASCADE,
    
    -- Indexes for performance
    INDEX idx_intent_feedback_user_intent (user_id, intent_key),
    INDEX idx_intent_feedback_scope_intent (scope, intent_key),
    INDEX idx_intent_feedback_created (created_at),
    INDEX idx_intent_feedback_message (message_id)
);

-- Feedback application tracking
CREATE TABLE IF NOT EXISTS feedback_applications (
    id VARCHAR(36) PRIMARY KEY,
    
    -- Query info
    query_intent_key VARCHAR(64) NOT NULL,
    query_text TEXT NOT NULL,
    user_id VARCHAR(128),
    
    -- Applied feedback
    applied_feedback_ids JSON DEFAULT '[]',
    
    -- Results
    original_doc_scores JSON DEFAULT '{}',   -- Before feedback
    adjusted_doc_scores JSON DEFAULT '{}',   -- After feedback
    rejected_doc_ids JSON DEFAULT '[]',      -- Blacklisted docs
    
    -- Metadata
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_feedback_applications_intent_user (query_intent_key, user_id),
    INDEX idx_feedback_applications_applied_at (applied_at)
);

-- Feedback clusters for similar intents
CREATE TABLE IF NOT EXISTS feedback_clusters (
    id VARCHAR(36) PRIMARY KEY,
    
    -- Cluster info
    cluster_name VARCHAR(255),
    center_embedding JSON NOT NULL,  -- Cluster centroid
    threshold FLOAT DEFAULT 0.8,     -- Similarity threshold
    
    -- Member intents
    intent_keys JSON DEFAULT '[]',
    
    -- Statistics
    member_count INTEGER DEFAULT 0,
    feedback_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User feedback metrics and reputation
CREATE TABLE IF NOT EXISTS user_feedback_metrics (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(128) UNIQUE NOT NULL,
    
    -- Quality metrics
    total_feedback_count INTEGER DEFAULT 0,
    verified_feedback_count INTEGER DEFAULT 0,
    spam_feedback_count INTEGER DEFAULT 0,
    
    -- Reputation
    trust_score FLOAT DEFAULT 1.0,  -- 0.0 to 2.0
    is_trusted BOOLEAN DEFAULT FALSE,
    is_blocked BOOLEAN DEFAULT FALSE,
    
    -- Rate limiting
    last_feedback_time TIMESTAMP WITH TIME ZONE,
    feedback_rate_per_hour FLOAT DEFAULT 0.0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_user_feedback_metrics_user_id (user_id),
    INDEX idx_user_feedback_metrics_trust_score (trust_score)
);

-- Add constraint to ensure chunks table only contains original content
-- (This is a logical constraint enforced by application logic)
-- ALTER TABLE chunks ADD CONSTRAINT chk_no_feedback_contamination 
-- CHECK (source != 'user_feedback' OR source = 'original');

-- Create a view to verify separation integrity
CREATE OR REPLACE VIEW separation_integrity_check AS
SELECT 
    'docs_index' as storage_type,
    COUNT(*) as total_items,
    COUNT(CASE WHEN source = 'original' THEN 1 END) as original_items,
    COUNT(CASE WHEN source = 'user_feedback' THEN 1 END) as feedback_contamination
FROM chunks
UNION ALL
SELECT 
    'feedback_store' as storage_type,
    COUNT(*) as total_items,
    COUNT(*) as original_items,  -- All feedback items are "original" in their context
    0 as feedback_contamination  -- No contamination possible in separated store
FROM intent_feedback;

-- Insert initial data or migration notes
INSERT INTO intent_keys (id, normalized_text, entities, tokens, embedding) 
VALUES ('migration_marker', 'separated_feedback_system_initialized', '[]', '[]', '[]')
ON DUPLICATE KEY UPDATE normalized_text = 'separated_feedback_system_initialized';

COMMIT;


