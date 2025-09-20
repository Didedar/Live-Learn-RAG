-- Enhanced feedback system migration
-- Adds new columns to existing tables and creates new tables for spam detection and content quality

-- Create user spam metrics table first
CREATE TABLE IF NOT EXISTS user_spam_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_identifier VARCHAR(255) UNIQUE NOT NULL,
    total_feedback_count INTEGER DEFAULT 0,
    spam_feedback_count INTEGER DEFAULT 0,
    last_feedback_time TIMESTAMP,
    feedback_rate_per_hour REAL DEFAULT 0.0,
    avg_confidence_score REAL DEFAULT 0.0,
    helpful_feedback_count INTEGER DEFAULT 0,
    reputation_score REAL DEFAULT 1.0,
    is_trusted BOOLEAN DEFAULT FALSE,
    is_blocked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create content quality metrics table
CREATE TABLE IF NOT EXISTS content_quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER UNIQUE NOT NULL,
    positive_feedback_count INTEGER DEFAULT 0,
    negative_feedback_count INTEGER DEFAULT 0,
    avg_rating REAL DEFAULT 0.0,
    total_ratings INTEGER DEFAULT 0,
    content_confidence REAL DEFAULT 0.5,
    factual_accuracy_score REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 0.5,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
);

-- Add new columns to feedback_events table (after creating other tables)
ALTER TABLE feedback_events ADD COLUMN rating INTEGER;
ALTER TABLE feedback_events ADD COLUMN user_ip VARCHAR(45);
ALTER TABLE feedback_events ADD COLUMN user_agent VARCHAR(500);
ALTER TABLE feedback_events ADD COLUMN session_fingerprint VARCHAR(128);
ALTER TABLE feedback_events ADD COLUMN is_spam BOOLEAN DEFAULT FALSE;
ALTER TABLE feedback_events ADD COLUMN is_filtered BOOLEAN DEFAULT FALSE;
ALTER TABLE feedback_events ADD COLUMN filter_reason VARCHAR(500);
ALTER TABLE feedback_events ADD COLUMN confidence_score REAL DEFAULT 0.0;

-- Create indexes for better performance (after all tables exist)
CREATE INDEX IF NOT EXISTS idx_user_spam_metrics_identifier ON user_spam_metrics(user_identifier);
CREATE INDEX IF NOT EXISTS idx_content_quality_metrics_chunk_id ON content_quality_metrics(chunk_id);
CREATE INDEX IF NOT EXISTS idx_feedback_events_user_ip ON feedback_events(user_ip);
CREATE INDEX IF NOT EXISTS idx_feedback_events_is_spam ON feedback_events(is_spam);
CREATE INDEX IF NOT EXISTS idx_feedback_events_is_filtered ON feedback_events(is_filtered);
CREATE INDEX IF NOT EXISTS idx_feedback_events_confidence_score ON feedback_events(confidence_score);

-- Update existing data with default values
UPDATE feedback_events SET 
    is_spam = FALSE,
    is_filtered = FALSE,
    confidence_score = 0.5
WHERE is_spam IS NULL OR is_filtered IS NULL OR confidence_score IS NULL;
