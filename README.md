# HN Signals

Extract actionable insights, pain points, and trends from Hacker News discussions using LLM.

## What it does

- Scrapes HN stories and comments (top, ask, show, best)
- Filters for relevant discussions using heuristics + LLM
- Extracts insights: pain points, feature requests, workflow problems, tool comparisons
- Generates markdown reports and JSON exports

## Setup

```bash
# Install dependencies
uv sync

# Set OpenAI API key
export OPENAI_API_KEY=your-key-here
```

## Usage

### Run full pipeline

```bash
uv run python main.py run
```

### Individual commands

```bash
# Scrape stories and comments
uv run python main.py scrape -t ask top -n 20

# Filter for relevant content
uv run python main.py filter --limit 50

# Extract insights
uv run python main.py extract --limit 30

# Analyze and generate reports
uv run python main.py analyze --report

# View stats
uv run python main.py stats
```

## Configuration

Edit `config.yaml` to set your interests and scraper settings:

```yaml
interests:
  - developer tools
  - API design
  - automation

scraper:
  story_types: [top, ask, show, best]
  stories_per_type: 30
```

## Output

Reports are saved in `reports/`:
- `insights_report_*.md` — Full analysis
- `daily_digest_*.md` — Summary
- `insights_export_*.json` — JSON export

## License

MIT
