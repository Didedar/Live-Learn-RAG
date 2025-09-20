-- Add status column to feedback_events table
-- Migration 004: Add status field to feedback_events

ALTER TABLE feedback_events ADD COLUMN status TEXT DEFAULT 'queued';

-- Update existing records to have a default status
UPDATE feedback_events SET status = 'applied' WHERE status IS NULL;
