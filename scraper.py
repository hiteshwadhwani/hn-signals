"""
Enhanced HN Scraper - Collects stories and comments from Hacker News API
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx
from loguru import logger

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"

# Story type endpoints
STORY_ENDPOINTS = {
    "top": "topstories",
    "new": "newstories",
    "best": "beststories",
    "ask": "askstories",
    "show": "showstories",
    "job": "jobstories",
}


@dataclass
class Story:
    """Represents a HN story"""

    id: int
    title: str
    url: Optional[str]
    text: Optional[str]
    by: str
    score: int
    time: int
    descendants: int  # comment count
    story_type: str
    kids: list[int]  # comment IDs

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.time)


@dataclass
class Comment:
    """Represents a HN comment"""

    id: int
    text: str
    by: str
    time: int
    parent: int
    story_id: int
    depth: int
    kids: list[int]
    deleted: bool = False
    dead: bool = False

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.time)


class HNScraper:
    """Async scraper for Hacker News API"""

    def __init__(
        self,
        timeout: float = 10.0,
        rate_limit_delay: float = 0.1,
        max_concurrent: int = 20,
    ):
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _get(self, endpoint: str) -> Optional[dict]:
        """Make a rate-limited GET request"""
        async with self._semaphore:
            try:
                await asyncio.sleep(self.rate_limit_delay)
                response = await self._client.get(f"{HN_API_BASE}/{endpoint}")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.warning(f"HTTP error fetching {endpoint}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error fetching {endpoint}: {e}")
                return None

    async def get_item(self, item_id: int) -> Optional[dict]:
        """Fetch a single item (story or comment)"""
        return await self._get(f"item/{item_id}.json")

    async def get_story_ids(self, story_type: str, limit: int = 30) -> list[int]:
        """Get story IDs for a given type"""
        endpoint = STORY_ENDPOINTS.get(story_type)
        if not endpoint:
            logger.error(f"Unknown story type: {story_type}")
            return []

        ids = await self._get(f"{endpoint}.json")
        if ids:
            return ids[:limit]
        return []

    async def fetch_story(self, story_id: int, story_type: str) -> Optional[Story]:
        """Fetch a story with its metadata"""
        item = await self.get_item(story_id)
        if not item or item.get("type") != "story":
            return None

        return Story(
            id=item["id"],
            title=item.get("title", ""),
            url=item.get("url"),
            text=item.get("text"),  # For Ask HN / Show HN posts
            by=item.get("by", "[deleted]"),
            score=item.get("score", 0),
            time=item.get("time", 0),
            descendants=item.get("descendants", 0),
            story_type=story_type,
            kids=item.get("kids", []),
        )

    async def fetch_comment(
        self,
        comment_id: int,
        story_id: int,
        depth: int = 0,
    ) -> Optional[Comment]:
        """Fetch a single comment"""
        item = await self.get_item(comment_id)
        if not item:
            return None

        # Handle deleted/dead comments
        if item.get("deleted") or item.get("dead"):
            return Comment(
                id=item["id"],
                text="",
                by="[deleted]",
                time=item.get("time", 0),
                parent=item.get("parent", 0),
                story_id=story_id,
                depth=depth,
                kids=[],
                deleted=item.get("deleted", False),
                dead=item.get("dead", False),
            )

        return Comment(
            id=item["id"],
            text=item.get("text", ""),
            by=item.get("by", "[deleted]"),
            time=item.get("time", 0),
            parent=item.get("parent", 0),
            story_id=story_id,
            depth=depth,
            kids=item.get("kids", []),
        )

    async def fetch_comment_tree(
        self,
        comment_ids: list[int],
        story_id: int,
        depth: int = 0,
        max_depth: int = 5,
    ) -> list[Comment]:
        """Recursively fetch all comments in a tree"""
        if depth > max_depth or not comment_ids:
            return []

        # Fetch all comments at this level concurrently
        tasks = [self.fetch_comment(cid, story_id, depth) for cid in comment_ids]
        results = await asyncio.gather(*tasks)

        comments = []
        child_tasks = []

        for comment in results:
            if comment and not comment.deleted and not comment.dead:
                comments.append(comment)
                if comment.kids:
                    child_tasks.append(
                        self.fetch_comment_tree(
                            comment.kids,
                            story_id,
                            depth + 1,
                            max_depth,
                        )
                    )

        # Fetch all child comment trees concurrently
        if child_tasks:
            child_results = await asyncio.gather(*child_tasks)
            for child_comments in child_results:
                comments.extend(child_comments)

        return comments

    async def fetch_stories(
        self,
        story_types: list[str],
        stories_per_type: int = 30,
        min_score: int = 0,
    ) -> list[Story]:
        """Fetch stories of specified types"""
        all_stories = []

        for story_type in story_types:
            logger.info(f"Fetching {story_type} stories...")
            story_ids = await self.get_story_ids(story_type, stories_per_type)

            tasks = [self.fetch_story(sid, story_type) for sid in story_ids]
            results = await asyncio.gather(*tasks)

            for story in results:
                if story and story.score >= min_score:
                    all_stories.append(story)

            logger.info(
                f"Fetched {len([s for s in results if s])} {story_type} stories"
            )

        return all_stories

    async def scrape_full(
        self,
        story_types: list[str],
        stories_per_type: int = 30,
        min_score: int = 0,
        max_comment_depth: int = 5,
        min_comments: int = 5,
    ) -> tuple[list[Story], list[Comment]]:
        """
        Full scrape: fetch stories and all their comments

        Returns:
            Tuple of (stories, comments)
        """
        logger.info("Starting full HN scrape...")

        # Fetch all stories
        stories = await self.fetch_stories(
            story_types,
            stories_per_type,
            min_score,
        )

        # Filter stories with enough comments
        stories_with_comments = [s for s in stories if s.descendants >= min_comments]

        logger.info(
            f"Found {len(stories_with_comments)} stories with "
            f">= {min_comments} comments"
        )

        # Fetch all comment trees
        all_comments = []
        for i, story in enumerate(stories_with_comments):
            logger.info(
                f"[{i + 1}/{len(stories_with_comments)}] "
                f"Fetching comments for: {story.title[:50]}..."
            )
            comments = await self.fetch_comment_tree(
                story.kids,
                story.id,
                max_depth=max_comment_depth,
            )
            all_comments.extend(comments)
            logger.info(f"  -> {len(comments)} comments fetched")

        logger.info(
            f"Scrape complete: {len(stories_with_comments)} stories, "
            f"{len(all_comments)} comments"
        )

        return stories_with_comments, all_comments


async def main():
    """Test the scraper"""
    async with HNScraper() as scraper:
        stories, comments = await scraper.scrape_full(
            story_types=["ask", "top"],
            stories_per_type=5,
            min_score=10,
            max_comment_depth=3,
            min_comments=5,
        )

        print(f"\n=== Scraped {len(stories)} stories ===")
        for story in stories[:3]:
            print(f"- [{story.score}] {story.title}")

        print(f"\n=== Sample comments ===")
        for comment in comments[:5]:
            text_preview = comment.text[:100].replace("\n", " ")
            print(f"- [depth={comment.depth}] {text_preview}...")


if __name__ == "__main__":
    asyncio.run(main())
