"""
Insight Extractor - Uses LLM to extract actionable insights from HN threads
"""

import json
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from database import Database, InsightRecord


class ExtractedInsight(BaseModel):
    """Single insight extracted from a thread"""

    type: str
    content: str
    evidence: str
    confidence: float
    category: str
    comment_id: Optional[int] = None


class ExtractionResult(BaseModel):
    """Result of extracting insights from a thread"""

    story_id: int
    insights: list[ExtractedInsight]
    summary: str
    opportunity_score: float


class InsightExtractor:
    """
    Extracts structured insights from HN threads using LLM
    """

    VALID_TYPES = {
        "pain_point",
        "feature_request",
        "workflow_problem",
        "tool_comparison",
        "willingness_to_pay",
        "market_gap",
    }

    VALID_CATEGORIES = {
        "developer_tools",
        "infrastructure",
        "productivity",
        "ai_ml",
        "saas",
        "consumer",
        "fintech",
        "healthcare",
        "education",
        "other",
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        config_path: str = "config.yaml",
    ):
        # Load config
        self.config = {}
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

        llm_config = self.config.get("llm", {})
        self.model = model or llm_config.get("extraction_model", "gpt-4o")
        self.max_tokens = llm_config.get("max_tokens", 2000)
        self.temperature = llm_config.get("temperature", 0.3)

        self.client = OpenAI()

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "extract.txt"
        if prompt_path.exists():
            self.prompt_template = prompt_path.read_text()
        else:
            logger.warning("Extract prompt not found, using default")
            self.prompt_template = self._default_prompt()

    def _default_prompt(self) -> str:
        return """Extract startup insights from this HN thread.

Title: {title}
Type: {story_type}
Score: {score}
URL: {url}

Story: {story_text}

Comments:
{comments}

Return JSON with insights array containing: type, content, evidence, confidence, category.
Also include: summary, opportunity_score (1-10)"""

    def _prepare_comments(
        self,
        comments: list[dict],
        max_chars: int = 12000,
    ) -> str:
        """
        Prepare comments for LLM input, respecting token limits
        """
        # Sort by depth then by position to maintain thread structure
        sorted_comments = sorted(
            comments, key=lambda c: (c.get("depth", 0), c.get("timestamp", 0))
        )

        formatted = []
        total_chars = 0

        for c in sorted_comments:
            text = c.get("text", "").strip()
            if not text:
                continue

            # Format: indent based on depth
            depth = c.get("depth", 0)
            indent = "  " * depth
            author = c.get("author", "anon")
            comment_id = c.get("id", "")

            line = f"{indent}[{author}] (id:{comment_id}): {text}"

            if total_chars + len(line) > max_chars:
                formatted.append("... (additional comments truncated)")
                break

            formatted.append(line)
            total_chars += len(line)

        return "\n\n".join(formatted)

    def extract(
        self,
        story_id: int,
        title: str,
        story_text: Optional[str],
        url: Optional[str],
        story_type: str,
        score: int,
        comments: list[dict],
    ) -> ExtractionResult:
        """
        Extract insights from a thread using LLM
        """
        # Prepare comments
        comments_text = self._prepare_comments(comments)

        # Format prompt
        prompt = self.prompt_template.format(
            title=title,
            story_type=story_type,
            score=score,
            url=url or "(no URL)",
            story_text=story_text or "(no text)",
            comments=comments_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract actionable startup insights from HN discussions. "
                            "Always respond with valid JSON matching the specified format."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Parse and validate insights
            insights = []
            for raw_insight in result.get("insights", []):
                # Validate type
                insight_type = raw_insight.get("type", "other")
                if insight_type not in self.VALID_TYPES:
                    insight_type = "pain_point"  # Default

                # Validate category
                category = raw_insight.get("category", "other")
                if category not in self.VALID_CATEGORIES:
                    category = "other"

                insights.append(
                    ExtractedInsight(
                        type=insight_type,
                        content=raw_insight.get("content", ""),
                        evidence=raw_insight.get("evidence", ""),
                        confidence=min(
                            1.0, max(0.0, raw_insight.get("confidence", 0.5))
                        ),
                        category=category,
                        comment_id=raw_insight.get("comment_id"),
                    )
                )

            extraction = ExtractionResult(
                story_id=story_id,
                insights=insights,
                summary=result.get("summary", ""),
                opportunity_score=min(10, max(0, result.get("opportunity_score", 5))),
            )

            logger.info(
                f"Extracted {len(insights)} insights from story {story_id}, "
                f"opportunity_score={extraction.opportunity_score}"
            )

            return extraction

        except Exception as e:
            logger.error(f"Extraction error for story {story_id}: {e}")
            return ExtractionResult(
                story_id=story_id,
                insights=[],
                summary=f"Extraction failed: {e}",
                opportunity_score=0,
            )

    def to_records(
        self,
        extraction: ExtractionResult,
    ) -> list[InsightRecord]:
        """
        Convert extraction result to database records
        """
        records = []
        for insight in extraction.insights:
            records.append(
                InsightRecord(
                    story_id=extraction.story_id,
                    comment_id=insight.comment_id,
                    insight_type=insight.type,
                    content=insight.content,
                    evidence=insight.evidence,
                    confidence=insight.confidence,
                    category=insight.category,
                )
            )
        return records


class BatchExtractor:
    """
    Extracts insights from multiple threads and saves to database
    """

    def __init__(
        self,
        db: Database,
        config_path: str = "config.yaml",
    ):
        self.db = db
        self.extractor = InsightExtractor(config_path=config_path)

    def process_story(self, story_id: int) -> ExtractionResult:
        """
        Process a single story: load from DB, extract insights, save results
        """
        # Load story
        story = self.db.get_story(story_id)
        if not story:
            logger.warning(f"Story {story_id} not found in database")
            return ExtractionResult(
                story_id=story_id,
                insights=[],
                summary="Story not found",
                opportunity_score=0,
            )

        # Load comments
        comments = self.db.get_comments_for_story(story_id)

        # Extract insights
        result = self.extractor.extract(
            story_id=story_id,
            title=story["title"],
            story_text=story.get("text"),
            url=story.get("url"),
            story_type=story["story_type"],
            score=story["score"],
            comments=comments,
        )

        # Save insights to database
        if result.insights:
            records = self.extractor.to_records(result)
            self.db.save_insights(records)
            logger.info(f"Saved {len(records)} insights for story {story_id}")

        # Mark story as processed
        self.db.mark_story_processed(story_id)

        return result

    def process_unprocessed(
        self,
        limit: int = 50,
        min_relevance: float = 5.0,
    ) -> list[ExtractionResult]:
        """
        Process all unprocessed stories that passed filtering
        """
        stories = self.db.get_unprocessed_stories(
            limit=limit,
            min_relevance=min_relevance,
        )

        logger.info(f"Processing {len(stories)} unprocessed stories")

        results = []
        for i, story in enumerate(stories):
            logger.info(
                f"[{i + 1}/{len(stories)}] Processing: {story['title'][:50]}..."
            )
            result = self.process_story(story["id"])
            results.append(result)

        return results


if __name__ == "__main__":
    # Test extraction with sample data
    extractor = InsightExtractor()

    sample_comments = [
        {
            "id": 1,
            "author": "user1",
            "depth": 0,
            "text": "I hate how every database migration tool is so complicated. Just want something simple that works.",
            "timestamp": 1,
        },
        {
            "id": 2,
            "author": "user2",
            "depth": 1,
            "text": "Same here. I've tried 5 different tools and they all have their own quirks.",
            "timestamp": 2,
        },
        {
            "id": 3,
            "author": "user3",
            "depth": 0,
            "text": "Would pay good money for a tool that just handles schema changes without breaking things.",
            "timestamp": 3,
        },
        {
            "id": 4,
            "author": "user4",
            "depth": 1,
            "text": "Every time I have to do a migration in production, I spend half a day just testing.",
            "timestamp": 4,
        },
    ]

    print("Testing insight extraction...")
    print("(This would call the OpenAI API - skipping actual call in test)")
