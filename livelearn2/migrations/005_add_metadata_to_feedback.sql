-- Add extra_data column to feedback_events table  
-- Migration 005: Add extra_data field for enhanced learning

ALTER TABLE feedback_events ADD COLUMN extra_data JSON DEFAULT '{}';

-- Update existing records to have empty extra_data
UPDATE feedback_events SET extra_data = '{}' WHERE extra_data IS NULL;
