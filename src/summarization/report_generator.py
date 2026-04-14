from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from src.config import Settings


class ReportGenerator:
    def __init__(self, settings: Settings) -> None:
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader("templates"))

    def save_markdown(self, content: str, date: datetime | None = None) -> str:
        if date is None:
            date = datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        filename = f"financial_briefing_{date_str}.md"
        filepath = self.output_dir / filename
        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    def _markdown_to_html(self, markdown: str) -> str:
        html_lines = []
        in_list = False

        for line in markdown.split("\n"):
            line = line.strip()

            if not line:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                continue

            if line.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h2>{line[3:]}</h2>")

            elif line.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h1>{line[2:]}</h1>")

            elif line.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{line[2:]}</li>")

            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<p>{line}</p>")

        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

    def generate_html(self, content: str, date: datetime | None = None) -> str:
        if date is None:
            date = datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        body_html = self._markdown_to_html(content)
        template = self.jinja_env.get_template("report.html")
        rendered = template.render(
            title="Daily Financial Briefing",
            date=date_str,
            content=body_html,
        )
        filename = f"financial_briefing_{date_str}.html"
        filepath = self.output_dir / filename
        filepath.write_text(rendered, encoding="utf-8")
        return str(filepath)


if __name__ == "__main__":
    settings = Settings()  # type: ignore
    rg = ReportGenerator(settings)

    sample_markdown = """## Major Market Movements
- S&P 500 closed up 1.2% at a new record high, per Reuters.
- Nasdaq gained 1.8% led by tech stocks.

## Federal Reserve & Monetary Policy
- Fed Chair Powell hinted at one more rate hike this year, per CNBC.

## Corporate Earnings & News
- Apple reported record Q2 revenue of $95B, per CNBC.

## Cryptocurrency & Digital Assets
- Bitcoin surged past $70,000 on ETF inflows, per Yahoo Finance.

## Key Themes of the Day
- Tech rally continues.
- Inflation concerns persist.

## Market Sentiment
Overall bullish sentiment driven by strong earnings.
"""

    md_path = rg.save_markdown(sample_markdown)
    print(f"Markdown saved: {md_path}")

    html_path = rg.generate_html(sample_markdown)
    print(f"HTML saved: {html_path}")
    