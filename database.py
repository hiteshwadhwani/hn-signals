"""
SQLite Storage Layer for HN Insights Scraper
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from scraper import Comment, Story


class InsightRecord(BaseModel):
    """Stored insight from LLM analysis"""

    id: Optional[int] = None
    story_id: int
    comment_id: Optional[int] = None
    insight_type: str  # pain_point, feature_request, workflow_problem, etc.
    content: str
    evidence: str  # The original text that led to this insight
    confidence: float  # 0-1 confidence score
    category: str  # Domain category
    created_at: Optional[datetime] = None


class ScrapeRun(BaseModel):
    """Track scraping runs for incremental updates"""

    id: Optional[int] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    story_count: int = 0
    comment_count: int = 0
    story_types: str  # Comma-separated


SCHEMA = """
-- Stories table
CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT,
    text TEXT,
    author TEXT NOT NULL,
    score INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    comment_count INTEGER NOT NULL,
    story_type TEXT NOT NULL,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    relevance_score REAL DEFAULT 0,
    processed BOOLEAN DEFAULT FALSE
);

-- Comments table
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY,
    story_id INTEGER NOT NULL,
    parent_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    author TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (story_id) REFERENCES stories(id)
);

-- Insights table
CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id INTEGER NOT NULL,
    comment_id INTEGER,
    insight_type TEXT NOT NULL,
    content TEXT NOT NULL,
    evidence TEXT NOT NULL,
    confidence REAL NOT NULL,
    category TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (story_id) REFERENCES stories(id),
    FOREIGN KEY (comment_id) REFERENCES comments(id)
);

-- Scrape runs table
CREATE TABLE IF NOT EXISTS scrape_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    story_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    story_types TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_stories_type ON stories(story_type);
CREATE INDEX IF NOT EXISTS idx_stories_score ON stories(score DESC);
CREATE INDEX IF NOT EXISTS idx_stories_processed ON stories(processed);
CREATE INDEX IF NOT EXISTS idx_comments_story ON comments(story_id);
CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_insights_story ON insights(story_id);
CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
"""


class Database:
    """SQLite database for storing HN data and insights"""

    def __init__(self, db_path: str = "data/hn_insights.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _connect(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ==================== Stories ====================

    def save_story(self, story: Story) -> bool:
        """Save a story to the database (upsert)"""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO stories 
                (id, title, url, text, author, score, timestamp, 
                 comment_count, story_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    story.id,
                    story.title,
                    story.url,
                    story.text,
                    story.by,
                    story.score,
                    story.time,
                    story.descendants,
                    story.story_type,
                ),
            )
            conn.commit()
            return True

    def save_stories(self, stories: list[Story]) -> int:
        """Bulk save stories"""
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stories 
                (id, title, url, text, author, score, timestamp, 
                 comment_count, story_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    (
                        s.id,
                        s.title,
                        s.url,
                        s.text,
                        s.by,
                        s.score,
                        s.time,
                        s.descendants,
                        s.story_type,
                    )
                    for s in stories
                ],
            )
            conn.commit()
            return len(stories)

    def get_story(self, story_id: int) -> Optional[dict]:
        """Get a story by ID"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stories WHERE id = ?", (story_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_unprocessed_stories(
        self, limit: int = 100, min_relevance: float = 0
    ) -> list[dict]:
        """Get stories that haven't been processed for insights"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM stories 
                WHERE processed = FALSE AND relevance_score >= ?
                ORDER BY relevance_score DESC, score DESC
                LIMIT ?
            """,
                (min_relevance, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_relevance_score(self, story_id: int, relevance_score: float):
        """Update a story's relevance score (after filtering)"""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE stories 
                SET relevance_score = ?
                WHERE id = ?
            """,
                (relevance_score, story_id),
            )
            conn.commit()

    def mark_story_processed(self, story_id: int):
        """Mark a story as processed (after insight extraction)"""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE stories 
                SET processed = TRUE
                WHERE id = ?
            """,
                (story_id,),
            )
            conn.commit()

    def get_stories_by_type(self, story_type: str, limit: int = 50) -> list[dict]:
        """Get stories filtered by type"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM stories 
                WHERE story_type = ?
                ORDER BY score DESC
                LIMIT ?
            """,
                (story_type, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def story_exists(self, story_id: int) -> bool:
        """Check if a story already exists"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM stories WHERE id = ?", (story_id,)
            ).fetchone()
            return row is not None

    # ==================== Comments ====================

    def save_comment(self, comment: Comment) -> bool:
        """Save a comment to the database"""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO comments
                (id, story_id, parent_id, text, author, timestamp, depth)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    comment.id,
                    comment.story_id,
                    comment.parent,
                    comment.text,
                    comment.by,
                    comment.time,
                    comment.depth,
                ),
            )
            conn.commit()
            return True

    def save_comments(self, comments: list[Comment]) -> int:
        """Bulk save comments"""
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO comments
                (id, story_id, parent_id, text, author, timestamp, depth)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    (c.id, c.story_id, c.parent, c.text, c.by, c.time, c.depth)
                    for c in comments
                ],
            )
            conn.commit()
            return len(comments)

    def get_comments_for_story(self, story_id: int) -> list[dict]:
        """Get all comments for a story"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM comments 
                WHERE story_id = ?
                ORDER BY timestamp ASC
            """,
                (story_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_comment_count(self, story_id: int) -> int:
        """Get comment count for a story"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM comments WHERE story_id = ?", (story_id,)
            ).fetchone()
            return row["cnt"] if row else 0

    # ==================== Insights ====================

    def save_insight(self, insight: InsightRecord) -> int:
        """Save an insight and return its ID"""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO insights
                (story_id, comment_id, insight_type, content, evidence, 
                 confidence, category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    insight.story_id,
                    insight.comment_id,
                    insight.insight_type,
                    insight.content,
                    insight.evidence,
                    insight.confidence,
                    insight.category,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def save_insights(self, insights: list[InsightRecord]) -> int:
        """Bulk save insights"""
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO insights
                (story_id, comment_id, insight_type, content, evidence, 
                 confidence, category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    (
                        i.story_id,
                        i.comment_id,
                        i.insight_type,
                        i.content,
                        i.evidence,
                        i.confidence,
                        i.category,
                    )
                    for i in insights
                ],
            )
            conn.commit()
            return len(insights)

    def get_insights(
        self,
        insight_type: Optional[str] = None,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[dict]:
        """Get insights with optional filters"""
        query = "SELECT * FROM insights WHERE confidence >= ?"
        params = [min_confidence]

        if insight_type:
            query += " AND insight_type = ?"
            params.append(insight_type)

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_insights_for_story(self, story_id: int) -> list[dict]:
        """Get all insights for a story"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM insights
                WHERE story_id = ?
                ORDER BY confidence DESC
            """,
                (story_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_insight_types(self) -> list[tuple[str, int]]:
        """Get insight types with counts"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT insight_type, COUNT(*) as count
                FROM insights
                GROUP BY insight_type
                ORDER BY count DESC
            """).fetchall()
            return [(row["insight_type"], row["count"]) for row in rows]

    def get_categories(self) -> list[tuple[str, int]]:
        """Get categories with counts"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM insights
                GROUP BY category
                ORDER BY count DESC
            """).fetchall()
            return [(row["category"], row["count"]) for row in rows]

    # ==================== Scrape Runs ====================

    def start_scrape_run(self, story_types: list[str]) -> int:
        """Start a new scrape run and return its ID"""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scrape_runs (started_at, story_types)
                VALUES (?, ?)
            """,
                (datetime.now(), ",".join(story_types)),
            )
            conn.commit()
            return cursor.lastrowid

    def complete_scrape_run(self, run_id: int, story_count: int, comment_count: int):
        """Mark a scrape run as complete"""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE scrape_runs
                SET completed_at = ?, story_count = ?, comment_count = ?
                WHERE id = ?
            """,
                (datetime.now(), story_count, comment_count, run_id),
            )
            conn.commit()

    def get_last_scrape_run(self) -> Optional[dict]:
        """Get the most recent scrape run"""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM scrape_runs
                ORDER BY started_at DESC
                LIMIT 1
            """).fetchone()
            return dict(row) if row else None

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get database statistics"""
        with self._connect() as conn:
            story_count = conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
            comment_count = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
            insight_count = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
            processed_count = conn.execute(
                "SELECT COUNT(*) FROM stories WHERE processed = TRUE"
            ).fetchone()[0]

            return {
                "stories": story_count,
                "comments": comment_count,
                "insights": insight_count,
                "processed_stories": processed_count,
                "unprocessed_stories": story_count - processed_count,
            }


if __name__ == "__main__":
    # Test the database
    db = Database()
    print("Database initialized!")
    print(f"Stats: {db.get_stats()}")
