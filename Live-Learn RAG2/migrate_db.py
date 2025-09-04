#!/usr/bin/env python3
"""
Database migration script for RAG system.
Usage: python migrate_db.py [command]

Commands:
  init     - Initialize database
  migrate  - Run migrations
  reset    - Reset database (WARNING: deletes all data)
  backup   - Create database backup
  stats    - Show database statistics
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import text

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.database import init_db, get_db_session, engine, backup_database, check_db_health
from app.models.documents import Document, Chunk
from app.models.feedback import MessageSession, FeedbackEvent, IndexMutation, ChunkWeight


def init_database():
    """Initialize database with all tables."""
    logger.info("Initializing database...")
    
    try:
        init_db()
        logger.success("Database initialized successfully!")
        
        # Show initial stats
        show_stats()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False
    
    return True


def run_migrations():
    """Run database migrations."""
    logger.info("Running database migrations...")
    
    try:
        with engine.connect() as conn:
            # Check current migration status
            result = conn.execute(text("""
                SELECT name FROM migrations ORDER BY id DESC LIMIT 1
            """))
            
            last_migration = result.fetchone()
            if last_migration:
                logger.info(f"Last migration: {last_migration[0]}")
            else:
                logger.info("No migrations applied yet")
            
            # Run init_db which handles migrations
            init_db()
            
        logger.success("Migrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
    
    return True


def reset_database():
    """Reset database (delete all data)."""
    logger.warning("This will DELETE ALL DATA in the database!")
    
    # Ask for confirmation
    confirm = input("Type 'YES' to confirm: ")
    if confirm != "YES":
        logger.info("Operation cancelled")
        return False
    
    try:
        # Create backup first
        backup_path = backup_database()
        logger.info(f"Backup created: {backup_path}")
        
        # Drop all tables
        from app.database import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped")
        
        # Recreate database
        init_db()
        logger.success("Database reset completed!")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False
    
    return True


def create_backup():
    """Create database backup."""
    logger.info("Creating database backup...")
    
    try:
        backup_path = backup_database()
        logger.success(f"Backup created: {backup_path}")
        
        # Show backup size
        size = os.path.getsize(backup_path)
        logger.info(f"Backup size: {size / 1024:.1f} KB")
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False
    
    return True


def show_stats():
    """Show database statistics."""
    logger.info("Database Statistics:")
    
    try:
        with get_db_session() as db:
            # Document stats
            doc_count = db.query(Document).count()
            logger.info(f"  📄 Documents: {doc_count}")
            
            # Chunk stats
            chunk_count = db.query(Chunk).count()
            chunks_with_embeddings = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]"
            ).count()
            
            logger.info(f"  🔧 Chunks: {chunk_count}")
            logger.info(f"  🎯 With embeddings: {chunks_with_embeddings}")
            
            if chunk_count > 0:
                embedding_coverage = chunks_with_embeddings / chunk_count * 100
                logger.info(f"  📊 Embedding coverage: {embedding_coverage:.1f}%")
            
            # Feedback stats
            session_count = db.query(MessageSession).count()
            feedback_count = db.query(FeedbackEvent).count()
            mutation_count = db.query(IndexMutation).count()
            weight_count = db.query(ChunkWeight).count()
            
            logger.info(f"  💬 Message sessions: {session_count}")
            logger.info(f"  👍 Feedback events: {feedback_count}")
            logger.info(f"  🔄 Index mutations: {mutation_count}")
            logger.info(f"  ⚖️  Chunk weights: {weight_count}")
        
        # Database file info
        if os.path.exists(settings.db_path):
            size = os.path.getsize(settings.db_path)
            logger.info(f"  💾 Database size: {size / 1024 / 1024:.2f} MB")
        
        # Health check
        health = check_db_health()
        status = health.get('status', 'unknown')
        logger.info(f"  ❤️  Health: {status}")
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return False
    
    return True


def optimize_database():
    """Optimize database performance."""
    logger.info("Optimizing database...")
    
    try:
        from app.database import optimize_database as db_optimize
        db_optimize()
        logger.success("Database optimization completed!")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return False
    
    return True


def cleanup_old_data(days: int = 90):
    """Clean up old data."""
    logger.info(f"Cleaning up data older than {days} days...")
    
    try:
        from app.database import cleanup_old_data as db_cleanup
        db_cleanup(days)
        logger.success("Data cleanup completed!")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False
    
    return True


def export_data(output_file: str):
    """Export data to JSON file."""
    logger.info(f"Exporting data to {output_file}...")
    
    try:
        import json
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "documents": [],
            "feedback_events": []
        }
        
        with get_db_session() as db:
            # Export documents and chunks
            documents = db.query(Document).all()
            
            for doc in documents:
                doc_data = {
                    "id": doc.id,
                    "uri": doc.uri,
                    "metadata": doc.doc_metadata,
                    "created_at": doc.created_at.isoformat(),
                    "chunks": []
                }
                
                for chunk in doc.chunks:
                    chunk_data = {
                        "id": chunk.id,
                        "ordinal": chunk.ordinal,
                        "content": chunk.content,
                        "source": chunk.source,
                        "version": chunk.version,
                        "created_at": chunk.created_at.isoformat(),
                        "has_embedding": bool(chunk.embedding and chunk.embedding != "[]")
                    }
                    doc_data["chunks"].append(chunk_data)
                
                export_data["documents"].append(doc_data)
            
            # Export feedback events
            feedback_events = db.query(FeedbackEvent).all()
            
            for event in feedback_events:
                event_data = {
                    "id": event.id,
                    "message_id": event.message_id,
                    "label": event.label.value,
                    "scope": event.scope.value,
                    "correction_text": event.correction_text,
                    "reason": event.reason,
                    "target_doc_id": event.target_doc_id,
                    "target_chunk_id": event.target_chunk_id,
                    "created_at": event.created_at.isoformat()
                }
                export_data["feedback_events"].append(event_data)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Data exported to {output_file}")
        logger.info(f"Exported {len(export_data['documents'])} documents")
        logger.info(f"Exported {len(export_data['feedback_events'])} feedback events")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Database migration tool for RAG system")
    parser.add_argument("command", choices=[
        "init", "migrate", "reset", "backup", "stats", "optimize", "cleanup", "export"
    ], help="Command to execute")
    
    parser.add_argument("--days", type=int, default=90, 
                       help="Days to keep for cleanup command")
    parser.add_argument("--output", type=str, default="rag_export.json",
                       help="Output file for export command")
    
    args = parser.parse_args()
    
    logger.info(f"RAG Database Migration Tool")
    logger.info(f"Database: {settings.db_path}")
    logger.info(f"Command: {args.command}")
    
    # Execute command
    success = False
    
    if args.command == "init":
        success = init_database()
    elif args.command == "migrate":
        success = run_migrations()
    elif args.command == "reset":
        success = reset_database()
    elif args.command == "backup":
        success = create_backup()
    elif args.command == "stats":
        success = show_stats()
    elif args.command == "optimize":
        success = optimize_database()
    elif args.command == "cleanup":
        success = cleanup_old_data(args.days)
    elif args.command == "export":
        success = export_data(args.output)
    
    if success:
        logger.success("Operation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Operation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()