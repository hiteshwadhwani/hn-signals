"""
Pattern Analyzer - Aggregates insights and identifies startup opportunities
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from database import Database

load_dotenv()


class OpportunityScores(BaseModel):
    """Scores for a startup opportunity"""

    pain_intensity: float
    frequency: float
    market_size: float
    monetization: float
    feasibility: float

    @property
    def total(self) -> float:
        """Calculate weighted total score"""
        weights = {
            "pain_intensity": 0.25,
            "frequency": 0.20,
            "market_size": 0.20,
            "monetization": 0.25,
            "feasibility": 0.10,
        }
        return sum(getattr(self, k) * v for k, v in weights.items())


class StartupOpportunity(BaseModel):
    """A synthesized startup opportunity"""

    title: str
    description: str
    target_user: str
    problem_statement: str
    existing_solutions: list[str]
    differentiation: str
    evidence: list[str]
    scores: OpportunityScores
    categories: list[str]
    next_steps: list[str]
    source_insights: list[int] = []  # Insight IDs

    @property
    def total_score(self) -> float:
        return self.scores.total


class Trend(BaseModel):
    """An emerging trend from HN discussions"""

    trend: str
    supporting_signals: list[str]
    insight_count: int = 0


class AnalysisResult(BaseModel):
    """Complete analysis result"""

    opportunities: list[StartupOpportunity]
    trends: list[Trend]
    meta_insights: str
    total_insights_analyzed: int
    categories_breakdown: dict[str, int]
    insight_types_breakdown: dict[str, int]


@dataclass
class InsightCluster:
    """A cluster of related insights"""

    theme: str
    insights: list[dict] = field(default_factory=list)
    categories: set[str] = field(default_factory=set)
    total_confidence: float = 0.0

    @property
    def avg_confidence(self) -> float:
        if not self.insights:
            return 0.0
        return self.total_confidence / len(self.insights)


class PatternAnalyzer:
    """
    Analyzes insights to identify patterns and synthesize opportunities
    """

    def __init__(
        self,
        db: Database,
        config_path: str = "config.yaml",
    ):
        self.db = db

        # Load config
        self.config = {}
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

        llm_config = self.config.get("llm", {})
        self.model = llm_config.get("extraction_model", "gpt-4o")
        self.client = OpenAI()

        # Load synthesis prompt
        prompt_path = Path(__file__).parent / "prompts" / "synthesize.txt"
        if prompt_path.exists():
            self.prompt_template = prompt_path.read_text()
        else:
            self.prompt_template = self._default_prompt()

    def _default_prompt(self) -> str:
        return """Synthesize these insights into startup opportunities:

{insights_json}

Return JSON with: opportunities (array), trends (array), meta_insights (string)"""

    def get_insights_summary(
        self,
        min_confidence: float = 0.5,
        limit: int = 200,
    ) -> dict:
        """
        Get a summary of insights for analysis
        """
        insights = self.db.get_insights(
            min_confidence=min_confidence,
            limit=limit,
        )

        # Breakdowns
        by_type = defaultdict(int)
        by_category = defaultdict(int)

        for insight in insights:
            by_type[insight["insight_type"]] += 1
            by_category[insight["category"]] += 1

        return {
            "total": len(insights),
            "by_type": dict(by_type),
            "by_category": dict(by_category),
            "insights": insights,
        }

    def cluster_insights(
        self,
        insights: list[dict],
    ) -> list[InsightCluster]:
        """
        Group related insights into clusters based on content similarity.
        Uses a simple keyword-based approach (can be enhanced with embeddings).
        """
        # For now, cluster by category and insight type
        clusters_map: dict[str, InsightCluster] = {}

        for insight in insights:
            key = f"{insight['category']}_{insight['insight_type']}"

            if key not in clusters_map:
                clusters_map[key] = InsightCluster(
                    theme=f"{insight['category'].replace('_', ' ').title()} - {insight['insight_type'].replace('_', ' ').title()}"
                )

            cluster = clusters_map[key]
            cluster.insights.append(insight)
            cluster.categories.add(insight["category"])
            cluster.total_confidence += insight["confidence"]

        # Sort clusters by total insight count and confidence
        clusters = list(clusters_map.values())
        clusters.sort(
            key=lambda c: (len(c.insights), c.avg_confidence),
            reverse=True,
        )

        return clusters

    def prepare_insights_for_llm(
        self,
        insights: list[dict],
        max_insights: int = 100,
    ) -> str:
        """
        Prepare insights for LLM synthesis, respecting token limits
        """
        # Prioritize high-confidence insights
        sorted_insights = sorted(
            insights,
            key=lambda x: x["confidence"],
            reverse=True,
        )[:max_insights]

        # Format for LLM
        formatted = []
        for i, insight in enumerate(sorted_insights):
            formatted.append(
                {
                    "id": insight.get("id"),
                    "type": insight["insight_type"],
                    "category": insight["category"],
                    "content": insight["content"],
                    "evidence": insight["evidence"],
                    "confidence": insight["confidence"],
                }
            )

        return json.dumps(formatted, indent=2)

    def synthesize_opportunities(
        self,
        insights: list[dict],
    ) -> AnalysisResult:
        """
        Use LLM to synthesize insights into startup opportunities
        """
        if not insights:
            logger.warning("No insights to analyze")
            return AnalysisResult(
                opportunities=[],
                trends=[],
                meta_insights="No insights available for analysis",
                total_insights_analyzed=0,
                categories_breakdown={},
                insight_types_breakdown={},
            )

        # Prepare insights
        insights_json = self.prepare_insights_for_llm(insights)

        # Calculate breakdowns
        by_type = defaultdict(int)
        by_category = defaultdict(int)
        for insight in insights:
            by_type[insight["insight_type"]] += 1
            by_category[insight["category"]] += 1

        # Format prompt
        prompt = self.prompt_template.format(insights_json=insights_json)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a startup strategist analyzing HN insights "
                            "to identify the best opportunities. Respond with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Parse opportunities
            opportunities = []
            for opp in result.get("opportunities", []):
                scores_data = opp.get("scores", {})
                scores = OpportunityScores(
                    pain_intensity=scores_data.get("pain_intensity", 5),
                    frequency=scores_data.get("frequency", 5),
                    market_size=scores_data.get("market_size", 5),
                    monetization=scores_data.get("monetization", 5),
                    feasibility=scores_data.get("feasibility", 5),
                )

                opportunities.append(
                    StartupOpportunity(
                        title=opp.get("title", "Untitled"),
                        description=opp.get("description", ""),
                        target_user=opp.get("target_user", ""),
                        problem_statement=opp.get("problem_statement", ""),
                        existing_solutions=opp.get("existing_solutions", []),
                        differentiation=opp.get("differentiation", ""),
                        evidence=opp.get("evidence", []),
                        scores=scores,
                        categories=opp.get("categories", []),
                        next_steps=opp.get("next_steps", []),
                    )
                )

            # Sort by total score
            opportunities.sort(key=lambda x: x.total_score, reverse=True)

            # Parse trends
            trends = []
            for t in result.get("trends", []):
                trends.append(
                    Trend(
                        trend=t.get("trend", ""),
                        supporting_signals=t.get("supporting_signals", []),
                    )
                )

            logger.info(
                f"Synthesized {len(opportunities)} opportunities, "
                f"{len(trends)} trends from {len(insights)} insights"
            )

            return AnalysisResult(
                opportunities=opportunities,
                trends=trends,
                meta_insights=result.get("meta_insights", ""),
                total_insights_analyzed=len(insights),
                categories_breakdown=dict(by_category),
                insight_types_breakdown=dict(by_type),
            )

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return AnalysisResult(
                opportunities=[],
                trends=[],
                meta_insights=f"Synthesis failed: {e}",
                total_insights_analyzed=len(insights),
                categories_breakdown=dict(by_category),
                insight_types_breakdown=dict(by_type),
            )

    def analyze(
        self,
        min_confidence: float = 0.5,
        max_insights: int = 200,
    ) -> AnalysisResult:
        """
        Full analysis pipeline: load insights, cluster, synthesize
        """
        logger.info("Starting pattern analysis...")

        # Get insights from database
        summary = self.get_insights_summary(
            min_confidence=min_confidence,
            limit=max_insights,
        )

        if not summary["insights"]:
            logger.warning("No insights in database")
            return AnalysisResult(
                opportunities=[],
                trends=[],
                meta_insights="No insights available",
                total_insights_analyzed=0,
                categories_breakdown={},
                insight_types_breakdown={},
            )

        logger.info(
            f"Analyzing {summary['total']} insights: "
            f"types={summary['by_type']}, categories={summary['by_category']}"
        )

        # Cluster insights
        clusters = self.cluster_insights(summary["insights"])
        logger.info(f"Created {len(clusters)} insight clusters")

        # Synthesize opportunities
        result = self.synthesize_opportunities(summary["insights"])

        return result

    def get_top_opportunities(
        self,
        n: int = 10,
        min_score: float = 6.0,
    ) -> list[StartupOpportunity]:
        """
        Get top N opportunities above a minimum score
        """
        result = self.analyze()

        filtered = [opp for opp in result.opportunities if opp.total_score >= min_score]

        return filtered[:n]


if __name__ == "__main__":
    print("Pattern Analyzer module loaded successfully")
    print("Run with actual database to analyze insights")
