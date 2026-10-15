"""Standalone interactive HTML report generator for EvalGuard AI.

Creates responsive, self-contained single-file dashboards with embedded SVG
radar charts, diagnostic breakdowns, and claim-level inspection.
"""

import html
import math
from typing import Dict, List
from evalguard.models import ClaimStatus, SuiteResult


class HTMLReporter:
    """Generates standalone HTML evaluation reports with embedded visual charts."""

    @staticmethod
    def _generate_svg_radar(metrics: Dict[str, float]) -> str:
        """Generates an embedded, pure SVG radar/spider chart for average metric scores."""
        if not metrics:
            return ""

        labels = list(metrics.keys())
        values = [max(0.0, min(1.0, metrics[k])) for k in labels]
        num_vars = len(labels)
        if num_vars < 3:
            return ""

        center_x = 220
        center_y = 200
        radius = 120

        # Background grid rings
        rings_svg = []
        for level in [0.25, 0.5, 0.75, 1.0]:
            r = radius * level
            points = []
            for i in range(num_vars):
                angle = (2 * math.pi / num_vars) * i - (math.pi / 2)
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append(f"{x:.1f},{y:.1f}")
            pts_str = " ".join(points)
            rings_svg.append(f'<polygon points="{pts_str}" fill="none" stroke="#334155" stroke-width="1" stroke-dasharray="3,3"/>')

        # Axis lines and labels
        axes_svg = []
        for i, label in enumerate(labels):
            angle = (2 * math.pi / num_vars) * i - (math.pi / 2)
            end_x = center_x + radius * math.cos(angle)
            end_y = center_y + radius * math.sin(angle)
            lbl_x = center_x + (radius + 24) * math.cos(angle)
            lbl_y = center_y + (radius + 24) * math.sin(angle)

            # Clean label formatting
            clean_label = label.replace("_", " ").title()
            axes_svg.append(f'<line x1="{center_x}" y1="{center_y}" x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="#475569" stroke-width="1.2"/>')
            anchor = "middle"
            if lbl_x < center_x - 10:
                anchor = "end"
            elif lbl_x > center_x + 10:
                anchor = "start"
            axes_svg.append(f'<text x="{lbl_x:.1f}" y="{lbl_y:.1f}" fill="#94a3b8" font-size="11" font-family="system-ui, sans-serif" text-anchor="{anchor}" alignment-baseline="middle">{clean_label}</text>')

        # Value polygon
        val_points = []
        for i, val in enumerate(values):
            angle = (2 * math.pi / num_vars) * i - (math.pi / 2)
            r = radius * val
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            val_points.append(f"{x:.1f},{y:.1f}")
        val_pts_str = " ".join(val_points)

        polygon_svg = f"""
        <polygon points="{val_pts_str}" fill="rgba(14, 165, 233, 0.35)" stroke="#0ea5e9" stroke-width="2.5"/>
        """

        # Dots on vertices
        dots_svg = []
        for pt in val_points:
            px, py = pt.split(",")
            dots_svg.append(f'<circle cx="{px}" cy="{py}" r="4.5" fill="#38bdf8" stroke="#0f172a" stroke-width="1.5"/>')

        return f"""
        <svg viewBox="0 0 440 400" width="100%" height="340" xmlns="http://www.w3.org/2000/svg">
            {''.join(rings_svg)}
            {''.join(axes_svg)}
            {polygon_svg}
            {''.join(dots_svg)}
        </svg>
        """

    @classmethod
    def render(cls, suite: SuiteResult, title: str = "EvalGuard AI - Evaluation Report") -> str:
        """Renders complete suite result into a self-contained, responsive HTML string."""
        sm = suite.summary
        radar_svg = cls._generate_svg_radar(sm.average_scores)

        pass_rate_pct = f"{sm.pass_rate * 100:.1f}%"
        status_color = "#10b981" if sm.pass_rate >= 0.8 else ("#f59e0b" if sm.pass_rate >= 0.5 else "#ef4444")

        # Build Test Case cards
        cards_html = []
        for r in suite.results:
            tc = r.test_case
            card_status_badge = (
                '<span class="badge badge-pass">PASSED</span>'
                if r.passed
                else '<span class="badge badge-fail">FAILED</span>'
            )

            # Metric rows
            metric_rows = []
            for m_name, m in r.metrics.items():
                m_st_badge = (
                    '<span class="badge badge-pass">PASS</span>'
                    if m.passed
                    else '<span class="badge badge-fail">FAIL</span>'
                )
                bar_pct = int(m.score * 100)
                metric_rows.append(f"""
                <tr>
                    <td class="font-medium">{html.escape(m_name.replace('_', ' ').title())}</td>
                    <td>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: {bar_pct}%;"></div>
                        </div>
                    </td>
                    <td class="font-mono">{m.score:.2f}</td>
                    <td class="font-mono text-muted">{m.threshold:.2f}</td>
                    <td>{m_st_badge}</td>
                    <td class="text-muted text-sm">{html.escape(m.reason or '')}</td>
                </tr>
                """)

            # Claims breakdown
            claims_html = ""
            if r.claims:
                claim_items = []
                for c in r.claims:
                    if c.status == ClaimStatus.SUPPORTED:
                        c_badge = '<span class="badge badge-pass">SUPPORTED</span>'
                    elif c.status == ClaimStatus.CONTRADICTED:
                        c_badge = '<span class="badge badge-fail">CONTRADICTED</span>'
                    else:
                        c_badge = '<span class="badge badge-warn">UNVERIFIED</span>'

                    evidence_html = (
                        f'<div class="evidence-box"><strong>Evidence:</strong> {html.escape(c.evidence)}</div>'
                        if c.evidence
                        else ""
                    )
                    claim_items.append(f"""
                    <div class="claim-item">
                        <div class="claim-header">{c_badge} <span class="claim-text">{html.escape(c.claim)}</span></div>
                        {evidence_html}
                    </div>
                    """)
                claims_html = f"""
                <div class="claims-section">
                    <h4>Claim-by-Claim Hallucination Audit ({len(r.claims)} claims)</h4>
                    {''.join(claim_items)}
                </div>
                """

            cards_html.append(f"""
            <div class="test-card">
                <div class="test-card-header">
                    <div class="test-id">
                        <span class="mono-id">{html.escape(tc.id)}</span>
                        {card_status_badge}
                    </div>
                    <div class="test-meta">Overall Score: <strong>{r.overall_score:.2f}</strong> | Eval Time: {r.execution_time_ms}ms</div>
                </div>
                <div class="test-body">
                    <div class="query-box">
                        <div class="label">USER QUERY</div>
                        <div class="content">{html.escape(tc.query)}</div>
                    </div>
                    <div class="output-box">
                        <div class="label">ACTUAL GENERATED OUTPUT</div>
                        <div class="content">{html.escape(tc.actual_output)}</div>
                    </div>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Score Bar</th>
                                <th>Score</th>
                                <th>Threshold</th>
                                <th>Status</th>
                                <th>Diagnostic Reason</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(metric_rows)}
                        </tbody>
                    </table>
                    {claims_html}
                </div>
            </div>
            """)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-dark: #090d16;
            --card-bg: #111827;
            --card-border: #1f2937;
            --primary: #0ea5e9;
            --secondary: #8b5cf6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 32px 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 32px;
        }}
        .logo-group h1 {{
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .logo-group p {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
        .header-meta {{ text-align: right; color: var(--text-muted); font-size: 13px; }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
        }}
        .kpi-label {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}
        .kpi-value {{ font-size: 32px; font-weight: 800; margin-top: 6px; }}

        .charts-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 36px;
        }}
        @media (max-width: 800px) {{ .charts-section {{ grid-template-columns: 1fr; }} }}
        .chart-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 24px;
        }}
        .chart-card h3 {{ font-size: 18px; margin-bottom: 16px; font-weight: 700; color: #e2e8f0; }}

        .test-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            margin-bottom: 24px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .test-card-header {{
            background: rgba(30, 41, 59, 0.5);
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
        }}
        .mono-id {{ font-family: monospace; font-weight: 700; font-size: 16px; color: var(--primary); }}
        .test-meta {{ color: var(--text-muted); font-size: 13px; }}
        .test-body {{ padding: 20px; }}

        .query-box, .output-box {{
            background: #0d131f;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 14px;
            border-left: 3px solid var(--primary);
        }}
        .output-box {{ border-left-color: var(--secondary); }}
        .query-box .label, .output-box .label {{ font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 4px; }}
        .query-box .content, .output-box .content {{ font-size: 14px; color: #e2e8f0; }}

        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 13px;
        }}
        .metrics-table th {{
            text-align: left;
            color: var(--text-muted);
            padding: 10px 12px;
            border-bottom: 1px solid var(--card-border);
            font-weight: 600;
        }}
        .metrics-table td {{
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .progress-bar-container {{
            width: 100px;
            height: 8px;
            background: #1e293b;
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-pass {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }}
        .badge-warn {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-yellow); border: 1px solid var(--accent-yellow); }}

        .claims-section {{
            margin-top: 20px;
            background: #0d131f;
            border-radius: 8px;
            padding: 16px;
        }}
        .claims-section h4 {{ font-size: 14px; margin-bottom: 12px; color: #cbd5e1; }}
        .claim-item {{ padding: 8px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); font-size: 13px; }}
        .claim-item:last-child {{ border-bottom: none; }}
        .claim-header {{ display: flex; align-items: center; gap: 8px; }}
        .claim-text {{ color: #e2e8f0; }}
        .evidence-box {{
            margin-top: 4px;
            margin-left: 95px;
            font-size: 12px;
            color: #94a3b8;
            background: rgba(30, 41, 59, 0.6);
            padding: 6px 10px;
            border-radius: 4px;
        }}
        footer {{ text-align: center; color: var(--text-muted); font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--card-border); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <h1>EvalGuard AI</h1>
                <p>Autonomous LLM & RAG Quality Assurance, Hallucination Detection & Guardrails</p>
            </div>
            <div class="header-meta">
                <div>Suite: <strong>{html.escape(suite.suite_name)}</strong></div>
                <div>Timestamp: {suite.timestamp}</div>
            </div>
        </header>

        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Pass Rate</div>
                <div class="kpi-value" style="color: {status_color};">{pass_rate_pct}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Tests</div>
                <div class="kpi-value">{sm.total_tests}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Passed / Failed</div>
                <div class="kpi-value"><span style="color: var(--accent-green);">{sm.passed_tests}</span> / <span style="color: var(--accent-red);">{sm.failed_tests}</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg Faithfulness</div>
                <div class="kpi-value">{sm.average_scores.get('faithfulness', 0.0):.2f}</div>
            </div>
        </section>

        <section class="charts-section">
            <div class="chart-card">
                <h3>Multi-Metric Radar Breakdown</h3>
                {radar_svg}
            </div>
            <div class="chart-card">
                <h3>Suite Health & Performance</h3>
                <div style="margin-top: 20px; font-size: 14px; line-height: 2;">
                    <div><strong>Average Answer Relevance:</strong> {sm.average_scores.get('answer_relevance', 0.0):.2f}</div>
                    <div><strong>Average Context Precision:</strong> {sm.average_scores.get('context_precision', 0.0):.2f}</div>
                    <div><strong>Average Context Recall:</strong> {sm.average_scores.get('context_recall', 0.0):.2f}</div>
                    <div><strong>Average Guardrail Safety:</strong> {sm.average_scores.get('guardrail_safety', 1.0):.2f}</div>
                    <div><strong>Execution Latency Avg:</strong> {f"{sm.average_latency_ms:.1f} ms" if sm.average_latency_ms else "N/A"}</div>
                    <div><strong>Total Tokens Processed:</strong> {sm.total_tokens or "N/A"}</div>
                </div>
            </div>
        </section>

        <section class="test-cases-section">
            <h2 style="font-size: 20px; font-weight: 700; margin-bottom: 20px; color: #f1f5f9;">Detailed Test Case Audits</h2>
            {''.join(cards_html)}
        </section>

        <footer>
            Generated by <strong>EvalGuard AI</strong> | Developed by Nitesh Beniwal
        </footer>
    </div>
</body>
</html>
"""
        return html_content

    @classmethod
    def save(cls, suite: SuiteResult, output_path: str) -> None:
        """Renders and writes HTML report to disk."""
        content = cls.render(suite)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
