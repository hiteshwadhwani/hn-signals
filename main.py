#!/usr/bin/env python3
"""
HN Insights Scraper - Main CLI Entry Point

Extract startup opportunities from Hacker News discussions.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from analyzer import PatternAnalyzer
from database import Database
from extractor import BatchExtractor
from filter import RelevanceFilter
from reporter import ReportGenerator
from scraper import HNScraper

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    if Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


async def cmd_scrape(args):
    """Scrape HN stories and comments"""
    config = load_config(args.config)
    scraper_config = config.get("scraper", {})
    db_config = config.get("database", {})

    db = Database(db_config.get("path", "data/hn_insights.db"))

    # Get settings
    story_types = args.types or scraper_config.get("story_types", ["top", "ask"])
    stories_per_type = args.count or scraper_config.get("stories_per_type", 30)
    min_score = args.min_score or scraper_config.get("min_story_score", 10)
    max_depth = scraper_config.get("max_comment_depth", 5)
    min_comments = config.get("filter", {}).get("min_comments", 5)

    console.print("[bold]Scraping HN[/bold]")
    console.print(f"  Story types: {', '.join(story_types)}")
    console.print(f"  Stories per type: {stories_per_type}")
    console.print(f"  Min score: {min_score}")
    console.print()

    # Start scrape run
    run_id = db.start_scrape_run(story_types)

    async with HNScraper(
        timeout=scraper_config.get("timeout", 10),
        rate_limit_delay=scraper_config.get("rate_limit_delay", 0.1),
    ) as scraper:
        stories, comments = await scraper.scrape_full(
            story_types=story_types,
            stories_per_type=stories_per_type,
            min_score=min_score,
            max_comment_depth=max_depth,
            min_comments=min_comments,
        )

    # Save to database
    if stories:
        db.save_stories(stories)
        console.print(f"[green]✓[/green] Saved {len(stories)} stories")

    if comments:
        db.save_comments(comments)
        console.print(f"[green]✓[/green] Saved {len(comments)} comments")

    # Complete scrape run
    db.complete_scrape_run(run_id, len(stories), len(comments))

    # Show stats
    stats = db.get_stats()
    console.print()
    console.print("[bold]Database Stats:[/bold]")
    console.print(f"  Total stories: {stats['stories']}")
    console.print(f"  Total comments: {stats['comments']}")
    console.print(f"  Unprocessed stories: {stats['unprocessed_stories']}")


async def cmd_filter(args):
    """Filter stories for relevance"""
    config = load_config(args.config)
    db_config = config.get("database", {})

    db = Database(db_config.get("path", "data/hn_insights.db"))
    relevance_filter = RelevanceFilter(config_path=args.config)

    # Get unprocessed stories
    stories = db.get_unprocessed_stories(limit=args.limit)

    if not stories:
        console.print("[yellow]No unprocessed stories found[/yellow]")
        return

    console.print(f"[bold]Filtering {len(stories)} stories[/bold]")
    console.print()

    passed_count = 0

    for i, story in enumerate(stories):
        comments = db.get_comments_for_story(story["id"])
        comment_texts = [c["text"] for c in comments if c.get("text")]

        result = relevance_filter.filter(
            story_id=story["id"],
            title=story["title"],
            story_text=story.get("text"),
            comments=comment_texts,
            story_type=story["story_type"],
            score=story["score"],
            comment_count=len(comments),
        )

        # Update story with relevance score (don't mark as processed yet)
        db.update_relevance_score(story["id"], result.relevance_score)

        status = "[green]PASS[/green]" if result.passed_llm else "[red]FAIL[/red]"
        console.print(
            f"[{i + 1}/{len(stories)}] {status} "
            f"(score={result.relevance_score:.1f}) {story['title'][:50]}..."
        )

        if result.passed_llm:
            passed_count += 1

    console.print()
    console.print(
        f"[bold]Results:[/bold] {passed_count}/{len(stories)} stories passed filtering"
    )


async def cmd_extract(args):
    """Extract insights from filtered stories"""
    config = load_config(args.config)
    db_config = config.get("database", {})
    filter_config = config.get("filter", {})

    db = Database(db_config.get("path", "data/hn_insights.db"))
    extractor = BatchExtractor(db, config_path=args.config)

    min_relevance = args.min_relevance or filter_config.get("min_relevance_score", 5)

    console.print("[bold]Extracting insights[/bold]")
    console.print(f"  Min relevance: {min_relevance}")
    console.print()

    results = extractor.process_unprocessed(
        limit=args.limit,
        min_relevance=min_relevance,
    )

    total_insights = sum(len(r.insights) for r in results)
    console.print()
    console.print(
        f"[green]✓[/green] Extracted {total_insights} insights from {len(results)} stories"
    )

    # Show stats
    stats = db.get_stats()
    console.print(f"[bold]Total insights in database:[/bold] {stats['insights']}")


async def cmd_analyze(args):
    """Analyze insights and generate opportunities"""
    config = load_config(args.config)
    db_config = config.get("database", {})

    db = Database(db_config.get("path", "data/hn_insights.db"))
    analyzer = PatternAnalyzer(db, config_path=args.config)
    reporter = ReportGenerator(config_path=args.config)

    console.print("[bold]Analyzing insights...[/bold]")
    console.print()

    result = analyzer.analyze(
        min_confidence=args.min_confidence,
        max_insights=args.max_insights,
    )

    if not result.opportunities:
        console.print(
            "[yellow]No opportunities found. Try running scrape and extract first.[/yellow]"
        )
        return

    # Display top opportunities
    console.print(
        f"[bold green]Found {len(result.opportunities)} opportunities![/bold green]"
    )
    console.print()

    for i, opp in enumerate(result.opportunities[: args.top]):
        console.print(
            f"[bold]{i + 1}. {opp.title}[/bold] (Score: {opp.total_score:.1f}/10)"
        )
        console.print(f"   {opp.description[:150]}...")
        console.print()

    # Generate reports
    if args.report:
        report_path = reporter.generate_full_report(result)
        console.print(f"[green]✓[/green] Generated report: {report_path}")

        json_path = reporter.generate_json_export(result)
        console.print(f"[green]✓[/green] Generated JSON export: {json_path}")

        digest_path = reporter.generate_daily_digest(result, db)
        console.print(f"[green]✓[/green] Generated daily digest: {digest_path}")


async def cmd_run(args):
    """Run the full pipeline: scrape -> filter -> extract -> analyze"""
    console.print("[bold cyan]═══ HN Insights Scraper - Full Pipeline ═══[/bold cyan]")
    console.print()

    # Step 1: Scrape
    console.print("[bold]Step 1/4: Scraping HN...[/bold]")
    await cmd_scrape(args)
    console.print()

    # Step 2: Filter
    console.print("[bold]Step 2/4: Filtering stories...[/bold]")
    args.limit = 100
    await cmd_filter(args)
    console.print()

    # Step 3: Extract
    console.print("[bold]Step 3/4: Extracting insights...[/bold]")
    args.min_relevance = None
    await cmd_extract(args)
    console.print()

    # Step 4: Analyze
    console.print("[bold]Step 4/4: Analyzing and generating report...[/bold]")
    args.min_confidence = 0.5
    args.max_insights = 200
    args.top = 10
    args.report = True
    await cmd_analyze(args)

    console.print()
    console.print("[bold green]═══ Pipeline Complete! ═══[/bold green]")


async def cmd_stats(args):
    """Show database statistics"""
    config = load_config(args.config)
    db_config = config.get("database", {})

    db = Database(db_config.get("path", "data/hn_insights.db"))
    stats = db.get_stats()

    table = Table(title="HN Insights Database Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Stories", str(stats["stories"]))
    table.add_row("Processed Stories", str(stats["processed_stories"]))
    table.add_row("Unprocessed Stories", str(stats["unprocessed_stories"]))
    table.add_row("Total Comments", str(stats["comments"]))
    table.add_row("Total Insights", str(stats["insights"]))

    console.print(table)

    # Show insight breakdown
    if stats["insights"] > 0:
        console.print()

        types_table = Table(title="Insights by Type")
        types_table.add_column("Type", style="cyan")
        types_table.add_column("Count", justify="right")

        for insight_type, count in db.get_insight_types():
            types_table.add_row(insight_type.replace("_", " ").title(), str(count))

        console.print(types_table)

        console.print()

        cat_table = Table(title="Insights by Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right")

        for category, count in db.get_categories():
            cat_table.add_row(category.replace("_", " ").title(), str(count))

        console.print(cat_table)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HN Insights Scraper - Extract startup opportunities from Hacker News",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scrape command
    scrape_parser = subparsers.add_parser(
        "scrape", help="Scrape HN stories and comments"
    )
    scrape_parser.add_argument(
        "-t",
        "--types",
        nargs="+",
        choices=["top", "new", "best", "ask", "show", "job"],
        help="Story types to scrape",
    )
    scrape_parser.add_argument(
        "-n",
        "--count",
        type=int,
        help="Number of stories per type",
    )
    scrape_parser.add_argument(
        "--min-score",
        type=int,
        help="Minimum story score",
    )

    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter stories for relevance")
    filter_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=100,
        help="Maximum stories to filter",
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract insights from stories"
    )
    extract_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=50,
        help="Maximum stories to process",
    )
    extract_parser.add_argument(
        "--min-relevance",
        type=float,
        help="Minimum relevance score",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze insights and generate report"
    )
    analyze_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum insight confidence",
    )
    analyze_parser.add_argument(
        "--max-insights",
        type=int,
        default=200,
        help="Maximum insights to analyze",
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top opportunities to show",
    )
    analyze_parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report files",
    )

    # Run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument(
        "-t",
        "--types",
        nargs="+",
        choices=["top", "new", "best", "ask", "show", "job"],
        help="Story types to scrape",
    )
    run_parser.add_argument(
        "-n",
        "--count",
        type=int,
        help="Number of stories per type",
    )
    run_parser.add_argument(
        "--min-score",
        type=int,
        help="Minimum story score",
    )
    run_parser.add_argument(
        "--skip-llm-filter",
        action="store_true",
        help="Skip LLM filtering to save costs",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run the appropriate command
    if args.command == "scrape":
        asyncio.run(cmd_scrape(args))
    elif args.command == "filter":
        asyncio.run(cmd_filter(args))
    elif args.command == "extract":
        asyncio.run(cmd_extract(args))
    elif args.command == "analyze":
        asyncio.run(cmd_analyze(args))
    elif args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args))


def cli():
    """Entry point for setuptools console_scripts"""
    main()


if __name__ == "__main__":
    main()
