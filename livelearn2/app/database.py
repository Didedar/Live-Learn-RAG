"""Database configuration and session management."""

import os
from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings
from .core.exceptions import DatabaseError
from fastapi import HTTPException


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


# Create database engine
engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={
        "check_same_thread": False,
        "timeout": 30,  # 30 second timeout
    },
    pool_pre_ping=True,  # Verify connections before use
    echo=settings.debug,  # Log SQL in debug mode
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except HTTPException:
        # это не ошибка БД — пробрасываем как есть
        raise
    except Exception as e:
        db.rollback()
        # это уже похоже на реальную проблему с БД
        logger.error(f"Database error: {e}")
        raise DatabaseError(f"Database operation failed: {e}") from e
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise DatabaseError(f"Database operation failed: {e}") from e
    finally:
        db.close()


def init_db():
    """
    Initialize database with tables and basic setup.
    """
    try:
        logger.info("Initializing database", db_path=settings.db_path)
        
        # Import all models to ensure they're registered
        from .models.documents import Document, Chunk  # noqa: F401
        from .models.feedback import (  # noqa: F401
            MessageSession,
            FeedbackEvent,
            IndexMutation,
            ChunkWeight
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Run initial migrations if needed
        run_migrations()
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise DatabaseError(f"Database initialization failed: {e}") from e


def run_migrations():
    """
    Run database migrations.
    
    This is a simple migration system. For production,
    consider using Alembic for more sophisticated migrations.
    """
    try:
        with engine.connect() as conn:
            # Check if migrations table exists
            result = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='migrations'
            """))
            
            if not result.fetchone():
                # Create migrations table
                conn.execute(text("""
                    CREATE TABLE migrations (
                        id INTEGER PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                logger.info("Created migrations table")
            
            # Get applied migrations
            result = conn.execute(text("SELECT name FROM migrations"))
            applied_migrations = {row[0] for row in result.fetchall()}
            
            # Define migration list
            migrations = [
                "001_initial_schema",
                "002_feedback_tables",
                "003_add_indexes",
                "004_chunk_weights",
            ]
            
            # Apply pending migrations
            for migration in migrations:
                if migration not in applied_migrations:
                    apply_migration(conn, migration)
                    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise DatabaseError(f"Migration failed: {e}") from e


def apply_migration(conn, migration_name: str):
    """Apply a specific migration."""
    try:
        logger.info(f"Applying migration: {migration_name}")
        
        if migration_name == "001_initial_schema":
            # Already handled by SQLAlchemy
            pass
            
        elif migration_name == "002_feedback_tables":
            # Already handled by SQLAlchemy
            pass
            
        elif migration_name == "003_add_indexes":
            # Add performance indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                ON chunks(document_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_not_null 
                ON chunks(id) WHERE embedding IS NOT NULL AND embedding != '[]'
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_feedback_events_message_id 
                ON feedback_events(message_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_index_mutations_status 
                ON index_mutations(status)
            """))
            
        elif migration_name == "004_chunk_weights":
            # Already handled by SQLAlchemy
            pass
        
        # Record migration as applied
        conn.execute(text("""
            INSERT INTO migrations (name) VALUES (:name)
        """), {"name": migration_name})
        
        conn.commit()
        logger.info(f"Migration {migration_name} applied successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to apply migration {migration_name}: {e}")
        raise


def check_db_health() -> dict:
    """
    Check database health and return status.
    
    Returns:
        Dictionary with health information
    """
    try:
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            
            # Get table counts
            stats = {}
            
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM documents"))
                stats["documents"] = result.fetchone()[0]
            except:
                stats["documents"] = "unknown"
            
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
                stats["chunks"] = result.fetchone()[0]
            except:
                stats["chunks"] = "unknown"
            
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM feedback_events"))
                stats["feedback_events"] = result.fetchone()[0]
            except:
                stats["feedback_events"] = "unknown"
            
            # Check database file size
            db_size = 0
            try:
                if os.path.exists(settings.db_path):
                    db_size = os.path.getsize(settings.db_path)
            except:
                pass
            
            return {
                "status": "healthy",
                "db_path": settings.db_path,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
                "tables": stats,
                "engine_pool_size": engine.pool.size(),
                "engine_pool_checked_in": engine.pool.checkedin(),
                "engine_pool_checked_out": engine.pool.checkedout(),
            }
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "db_path": settings.db_path
        }


def optimize_database():
    """
    Optimize database performance.
    """
    try:
        logger.info("Optimizing database")
        
        with engine.connect() as conn:
            # Analyze tables for better query planning
            conn.execute(text("ANALYZE"))
            
            # Vacuum to reclaim space and defragment
            conn.execute(text("VACUUM"))
            
            # Update SQLite settings for better performance
            conn.execute(text("PRAGMA journal_mode = WAL"))
            conn.execute(text("PRAGMA synchronous = NORMAL"))
            conn.execute(text("PRAGMA cache_size = 10000"))
            conn.execute(text("PRAGMA temp_store = memory"))
            
            conn.commit()
            
        logger.info("Database optimization completed")
        
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        raise DatabaseError(f"Database optimization failed: {e}") from e


def backup_database(backup_path: str = None):
    """
    Create a backup of the database.
    
    Args:
        backup_path: Path for backup file
    """
    try:
        if backup_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{settings.db_path}.backup_{timestamp}"
        
        logger.info(f"Creating database backup: {backup_path}")
        
        # SQLite backup using built-in backup API
        import sqlite3
        
        # Connect to source and destination
        source = sqlite3.connect(settings.db_path)
        backup = sqlite3.connect(backup_path)
        
        # Perform backup
        source.backup(backup)
        
        # Close connections
        backup.close()
        source.close()
        
        logger.info(f"Database backup created successfully: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise DatabaseError(f"Database backup failed: {e}") from e


def cleanup_old_data(days: int = 90):
    """
    Clean up old data from the database.
    
    Args:
        days: Number of days to keep data
    """
    try:
        logger.info(f"Cleaning up data older than {days} days")
        
        with get_db_session() as db:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old message sessions
            result = db.execute(text("""
                DELETE FROM message_sessions 
                WHERE created_at < :cutoff_date
            """), {"cutoff_date": cutoff_date})
            
            deleted_messages = result.rowcount
            
            # Clean up orphaned feedback events
            result = db.execute(text("""
                DELETE FROM feedback_events 
                WHERE message_id NOT IN (SELECT id FROM message_sessions)
            """))
            
            deleted_feedback = result.rowcount
            
            logger.info(
                "Data cleanup completed",
                deleted_messages=deleted_messages,
                deleted_feedback=deleted_feedback
            )
            
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise DatabaseError(f"Data cleanup failed: {e}") from e