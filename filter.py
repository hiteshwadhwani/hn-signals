"""
Two-stage Relevance Filter for HN threads

Stage 1: Heuristic-based fast filtering
Stage 2: LLM-based relevance scoring
"""

import json
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel


class FilterResult(BaseModel):
    """Result of filtering a thread"""

    story_id: int
    relevance_score: float
    reasoning: str
    key_signals: list[str]
    categories: list[str]
    passed_heuristic: bool
    passed_llm: bool


class LLMFilter:
    """
    Stage 2: LLM-based relevance filtering

    Uses a cheap LLM to score thread relevance based on configured interests.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        interests: Optional[list[str]] = None,
    ):
        self.model = model
        self.interests = interests or []
        self.client = OpenAI()

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "filter.txt"
        if prompt_path.exists():
            self.prompt_template = prompt_path.read_text()
        else:
            logger.warning("Filter prompt not found, using default")
            self.prompt_template = self._default_prompt()

    def _default_prompt(self) -> str:
        return """Analyze this HN thread for startup opportunity relevance.
        
Interests: {interests}
Title: {title}
Type: {story_type}
Score: {score}
Comments: {comment_count}

Story: {story_text}

Sample Comments:
{comments}

Return JSON: {{"relevance_score": 1-10, "reasoning": "...", "key_signals": [...], "categories": [...]}}"""

    def filter_thread(
        self,
        story_id: int,
        title: str,
        story_text: Optional[str],
        comments: list[str],
        story_type: str,
        score: int,
        comment_count: int,
        min_score: int = 5,
    ) -> FilterResult:
        """
        Use LLM to score thread relevance.
        """
        # Prepare comments summary (limit tokens)
        comments_text = "\n---\n".join(comments[:20])
        if len(comments_text) > 4000:
            comments_text = comments_text[:4000] + "\n... (truncated)"

        # Format prompt
        prompt = self.prompt_template.format(
            interests=", ".join(self.interests),
            title=title,
            story_type=story_type,
            score=score,
            comment_count=comment_count,
            story_text=story_text or "(No text)",
            comments=comments_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze HN threads for startup insights. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            relevance_score = result.get("relevance_score", 0)
            passed = relevance_score >= min_score

            logger.info(
                f"LLM filter: story={story_id}, score={relevance_score}, "
                f"passed={passed}"
            )

            return FilterResult(
                story_id=story_id,
                relevance_score=relevance_score,
                reasoning=result.get("reasoning", ""),
                key_signals=result.get("key_signals", []),
                categories=result.get("categories", []),
                passed_heuristic=True,
                passed_llm=passed,
            )

        except Exception as e:
            logger.error(f"LLM filter error: {e}")
            # On error, be conservative and pass the thread
            return FilterResult(
                story_id=story_id,
                relevance_score=5.0,
                reasoning=f"LLM error: {e}",
                key_signals=[],
                categories=[],
                passed_heuristic=True,
                passed_llm=True,
            )


class RelevanceFilter:
    """
    Combined two-stage filter
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        config = {}
        if Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        # Initialize filters
        filter_config = config.get("filter", {})
        llm_config = config.get("llm", {})

        self.llm = LLMFilter(
            model=llm_config.get("filter_model", "gpt-4o-mini"),
            interests=config.get("interests", []),
        )

        self.llm_threshold = filter_config.get("min_relevance_score", 5)
        self.min_comments = filter_config.get("min_comments", 5)

    def filter(
        self,
        story_id: int,
        title: str,
        story_text: Optional[str],
        comments: list[str],
        story_type: str,
        score: int,
        comment_count: int,
    ) -> FilterResult:
        """
        Run two-stage filtering on a thread.

        Args:
            skip_llm: If True, only run heuristic filter (for testing/cost saving)

        Returns:
            FilterResult with relevance score and signals
        """
        # Check minimum comments
        if comment_count < self.min_comments:
            return FilterResult(
                story_id=story_id,
                relevance_score=0,
                reasoning="Too few comments",
                key_signals=[],
                categories=[],
                passed_heuristic=False,
                passed_llm=False,
            )

        result = self.llm.filter_thread(
            story_id=story_id,
            title=title,
            story_text=story_text,
            comments=comments,
            story_type=story_type,
            score=score,
            comment_count=comment_count,
            min_score=self.llm_threshold,
        )

        result.key_signals = list(set(result.key_signals))

        return result
