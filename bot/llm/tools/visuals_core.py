"""
visuals_core.py
Standalone visualization tool for OllamaService.

Entry point: generate_visualization(**kwargs) -> str
Supports: table, comparison_table, chart, multi_chart, heatmap, timeline, flowchart, tree
Output: Markdown/ASCII text (Discord-safe, no HTML/CDN dependencies)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Set, Tuple

# ── Constants ────────────────────────────────────────────────────────────────

MAX_ROWS = 250
MAX_COLS = 60
CHART_HEIGHT = 12
ChartType = Literal["bar", "line", "scatter"]


# ── Entry Point ──────────────────────────────────────────────────────────────


def generate_visualization(
    viz_type: str,
    title: str = "",
    **kwargs: Any,
) -> str:
    """
    Main entry point for OllamaService tool_map.

    viz_type options:
      "table"            - rows (list of dicts)
      "comparison_table" - items, criteria, scores
      "chart"            - x, y, chart_type (bar|line|scatter)
      "multi_chart"      - series (list of {name, x, y}), chart_type
      "heatmap"          - data, row_labels, col_labels
      "timeline"         - events (list of {date, label, detail})
      "flowchart"        - steps (list of {from, to, label}), direction (lr|tb)
      "tree"             - tree (dict or list of {parent, child} edges)
    """
    handlers = {
        "table": _render_table,
        "comparison_table": _render_comparison_table,
        "chart": _render_chart,
        "multi_chart": _render_multi_chart,
        "heatmap": _render_heatmap,
        "timeline": _render_timeline,
        "flowchart": _render_flowchart,
        "tree": _render_tree,
    }
    fn = handlers.get(viz_type)
    if not fn:
        return f"Unknown viz_type '{viz_type}'. Options: {', '.join(handlers)}"
    return fn(title=title, **kwargs)


# ── JSON Coercion ────────────────────────────────────────────────────────────


def _coerce(value: Any, expect: Literal["list", "dict", "any"]) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        s = value.strip().strip("\"'")
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except Exception:
                pass
    return value


def _str_list(value: Any) -> List[str]:
    v = _coerce(value, "list")
    return [str(x) for x in v] if isinstance(v, list) else ([str(value)] if value else [])


def _float_list(value: Any) -> List[float]:
    v = _coerce(value, "list")
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            pass
    return out


# ── Markdown Helpers ─────────────────────────────────────────────────────────


def _codeblock(title: str, text: str) -> str:
    header = f"### {title}\n\n" if title else ""
    return f"{header}```\n{text.rstrip()}\n```"


def _md_table(title: str, cols: List[str], rows: List[Dict[str, Any]]) -> str:
    def cell(v: Any) -> str:
        return str("" if v is None else v).replace("\n", " ").strip()

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join(
        "| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |" for r in rows
    )
    table = f"{header}\n{sep}\n{body}"
    return f"### {title}\n\n{table}" if title else table


# ── Tables ───────────────────────────────────────────────────────────────────


def _render_table(*, title: str = "", rows: Any = None, **_: Any) -> str:
    rows = _coerce(rows, "list")
    if not isinstance(rows, list) or not rows:
        return "No valid rows provided."
    rows = [r for r in rows if isinstance(r, dict)][:MAX_ROWS]
    cols: List[str] = []
    seen: Set[str] = set()
    for r in rows:
        for k in r:
            if str(k) not in seen:
                seen.add(str(k))
                cols.append(str(k))
    return _md_table(title, cols[:MAX_COLS], rows)


def _render_comparison_table(
    *,
    title: str = "",
    items: Any = None,
    criteria: Any = None,
    scores: Any = None,
    highlight_best: bool = True,
    **_: Any,
) -> str:
    items_l = _str_list(items)
    criteria_l = _str_list(criteria)
    scores_d = _coerce(scores, "dict")
    if not isinstance(scores_d, dict):
        scores_d = {}

    cols = ["Item"] + criteria_l
    rows: List[Dict[str, Any]] = []
    best: Dict[str, float] = {}

    if highlight_best:
        for c in criteria_l:
            vals = []
            for item in items_l:
                v = scores_d.get(item, {}).get(c)
                try:
                    vals.append(float(v))  # type: ignore
                except (TypeError, ValueError):
                    pass
            if vals:
                best[c] = max(vals)

    for item in items_l:
        row: Dict[str, Any] = {"Item": item}
        item_scores = scores_d.get(item, {})
        for c in criteria_l:
            v = item_scores.get(c, "—")
            try:
                row[c] = (
                    f"{v} ★" if highlight_best and float(v) == best.get(c) else v  # type: ignore
                )
            except (TypeError, ValueError):
                row[c] = v
        rows.append(row)

    return _md_table(title or "Comparison", cols, rows)


# ── ASCII Charts ─────────────────────────────────────────────────────────────


def _x_labels(x: List[Any], n: int) -> str:
    return " ".join(f"{str(v)[:3]:^3}" for v in x) + " " * max(0, (n - len(x)) * 4)


def _ascii_bar(x: List[Any], y: List[float], title: str) -> str:
    if not y:
        return _codeblock(title, "No data.")
    h, max_y, min_y = CHART_HEIGHT, max(y), min(y)
    rng = max_y - min_y or 1.0
    lines = []
    for i in range(h, 0, -1):
        thr = min_y + rng * i / h
        row = "".join("█ " if v >= thr else "  " for v in y)
        lines.append(f"{thr:6.1f} │{row}")
    lines.append(" " * 8 + "┴" * (len(y) * 2))
    lines.append(" " * 8 + _x_labels(x, len(y)))
    return _codeblock(title, "\n".join(lines))


def _ascii_line(x: List[Any], y: List[float], title: str) -> str:
    if not y:
        return _codeblock(title, "No data.")
    h, max_y, min_y = CHART_HEIGHT, max(y), min(y)
    rng = max_y - min_y or 1.0
    grid = [[" "] * (len(y) * 2) for _ in range(h)]

    for i, v in enumerate(y):
        row = max(0, min(h - 1, int((max_y - v) / rng * (h - 1))))
        grid[row][i * 2] = "●"
        if i > 0:
            pr = max(0, min(h - 1, int((max_y - y[i - 1]) / rng * (h - 1))))
            for r in range(min(pr, row), max(pr, row) + 1):
                if r != pr and r != row:
                    grid[r][i * 2 - 1] = "│"
            if pr == row:
                grid[row][i * 2 - 1] = "─"

    lines = []
    for i, row in enumerate(grid):
        v = max_y - rng * i / (h - 1) if h > 1 else max_y
        lines.append(f"{v:6.1f} │{''.join(row)}")
    lines.append(" " * 8 + "┴" * (len(y) * 2))
    lines.append(" " * 8 + _x_labels(x, len(y)))
    return _codeblock(title, "\n".join(lines))


def _ascii_scatter(x: List[Any], y: List[float], title: str) -> str:
    if not y:
        return _codeblock(title, "No data.")
    h, max_y, min_y = CHART_HEIGHT, max(y), min(y)
    rng = max_y - min_y or 1.0
    grid = [[" "] * (len(y) * 2) for _ in range(h)]
    for i, v in enumerate(y):
        row = max(0, min(h - 1, int((max_y - v) / rng * (h - 1))))
        grid[row][i * 2] = "●"
    lines = []
    for i, row in enumerate(grid):
        v = max_y - rng * i / (h - 1) if h > 1 else max_y
        lines.append(f"{v:6.1f} │{''.join(row)}")
    lines.append(" " * 8 + "┴" * (len(y) * 2))
    lines.append(" " * 8 + _x_labels(x, len(y)))
    return _codeblock(title, "\n".join(lines))


def _draw_chart(x: List[Any], y: List[float], title: str, chart_type: str) -> str:
    fn = {"bar": _ascii_bar, "scatter": _ascii_scatter}.get(chart_type, _ascii_line)
    return fn(x, y, title)


def _render_chart(
    *,
    title: str = "",
    x: Any = None,
    y: Any = None,
    chart_type: str = "line",
    **_: Any,
) -> str:
    x_l, y_l = _coerce(x, "list"), _float_list(y)
    if not isinstance(x_l, list):
        return "Invalid 'x': expected a list."
    if len(x_l) != len(y_l):
        return f"x and y length mismatch ({len(x_l)} vs {len(y_l)})."
    return _draw_chart(x_l, y_l, title or "Chart", chart_type)


def _render_multi_chart(
    *,
    title: str = "",
    series: Any = None,
    chart_type: str = "line",
    **_: Any,
) -> str:
    series_l = _coerce(series, "list")
    if not isinstance(series_l, list) or not series_l:
        return "Invalid 'series': expected a non-empty list."
    parts = [f"### {title}"] if title else []
    for s in series_l:
        if not isinstance(s, dict):
            continue
        x_l = _coerce(s.get("x", []), "list") or []
        y_l = _float_list(s.get("y", []))
        name = str(s.get("name", "Series"))
        parts.append(_draw_chart(x_l, y_l, name, chart_type))
    return "\n\n".join(parts)


# ── Heatmap ──────────────────────────────────────────────────────────────────


def _render_heatmap(
    *,
    title: str = "",
    data: Any = None,
    row_labels: Any = None,
    col_labels: Any = None,
    **_: Any,
) -> str:
    data_v = _coerce(data, "list")
    if not isinstance(data_v, list):
        return "Invalid 'data': expected a 2D list."

    matrix: List[List[float]] = []
    for row in data_v:
        row_l = _coerce(row, "list")
        if not isinstance(row_l, list):
            continue
        matrix.append(_float_list(row_l))

    if not matrix:
        return "No valid heatmap data."

    r = _str_list(row_labels) or [f"Row {i+1}" for i in range(len(matrix))]
    c = _str_list(col_labels) or [
        f"Col {j+1}" for j in range(max(len(row) for row in matrix))
    ]

    all_vals = [v for row in matrix for v in row]
    min_v, max_v = min(all_vals, default=0.0), max(all_vals, default=1.0)
    rng = max_v - min_v or 1.0
    blocks = " ░▒▓█"

    lines = [" " * 12 + " ".join(f"{str(col)[:8]:^8}" for col in c), ""]
    for i, rl in enumerate(r[: len(matrix)]):
        row_str = f"{str(rl)[:10]:10} │"
        for j in range(len(c)):
            v = matrix[i][j] if j < len(matrix[i]) else 0.0
            idx = min(int((v - min_v) / rng * (len(blocks) - 1)), len(blocks) - 1)
            row_str += f" {blocks[idx] * 2}  "
        lines.append(row_str)

    lines += ["", f"Range: {min_v:.2f} – {max_v:.2f}"]
    return _codeblock(title or "Heatmap", "\n".join(lines))


# ── Timeline ─────────────────────────────────────────────────────────────────


def _render_timeline(
    *,
    title: str = "",
    events: Any = None,
    show_dates: bool = True,
    **_: Any,
) -> str:
    ev = _coerce(events, "list")
    if not isinstance(ev, list) or not ev:
        return "No events provided."
    lines = []
    for e in ev[:MAX_ROWS]:
        if not isinstance(e, dict):
            continue
        d = str(e.get("date", "")).strip()
        lab = str(e.get("label", "Event")).strip()
        det = str(e.get("detail", "")).strip()
        prefix = f"{d} — " if show_dates and d else ""
        lines.append(f"{prefix}{lab}" + (f" :: {det}" if det else ""))
    return _codeblock(title or "Timeline", "\n".join(lines))


# ── Flowchart ────────────────────────────────────────────────────────────────


def _render_flowchart(
    *,
    title: str = "",
    steps: Any = None,
    direction: str = "tb",
    **_: Any,
) -> str:
    v = _coerce(steps, "list")
    if not isinstance(v, list):
        return "Invalid 'steps': expected a list of {from, to, label} objects."

    edges: List[Tuple[str, str, str]] = []
    for s in v[:MAX_ROWS]:
        if not isinstance(s, dict):
            continue
        a, b = str(s.get("from", "")).strip(), str(s.get("to", "")).strip()
        if a and b:
            edges.append((a, b, str(s.get("label", "")).strip()))

    if not edges:
        return "No valid edges. Each step needs 'from' and 'to'."

    if direction == "lr":
        order, seen = [], set()
        for a, b, _ in edges:
            for n in (a, b):
                if n not in seen:
                    seen.add(n)
                    order.append(n)
        parts = []
        for i, node in enumerate(order):
            parts.append(f"[{node}]")
            if i < len(order) - 1:
                lbl = next(
                    (
                        l
                        for a, b, l in edges
                        if a == node and b == order[i + 1] and l
                    ),
                    "",
                )
                parts.append(f"--{lbl}-->" if lbl else "---->")
        text = " ".join(parts)
    else:
        text = "\n".join(
            f"[{a}]{f' --{l}--> ' if l else ' --> '}[{b}]"
            for a, b, l in edges
        )

    return _codeblock(title or "Flowchart", text)


# ── Tree ─────────────────────────────────────────────────────────────────────


def _render_tree(
    *,
    title: str = "",
    tree: Any = None,
    root_label: str = "root",
    max_depth: int = 12,
    **_: Any,
) -> str:
    max_depth = max(1, min(int(max_depth), 50))
    v = _coerce(tree, "any")

    def from_dict(node: Any, prefix: str = "", depth: int = 0) -> List[str]:
        if depth > max_depth:
            return [prefix + "…"]
        out: List[str] = []
        if isinstance(node, dict):
            items = list(node.items())
            for i, (k, val) in enumerate(items):
                last = i == len(items) - 1
                out.append(prefix + ("└─ " if last else "├─ ") + str(k))
                out.extend(
                    from_dict(val, prefix + ("   " if last else "│  "), depth + 1)
                )
        elif isinstance(node, list):
            for i, val in enumerate(node):
                last = i == len(node) - 1
                branch = "└─ " if last else "├─ "
                if isinstance(val, (dict, list)):
                    out.extend(from_dict(val, prefix + branch, depth + 1))
                else:
                    out.append(prefix + branch + str(val))
        else:
            out.append(prefix + str(node))
        return out

    def from_edges(edges: List[Dict[str, Any]]) -> str:
        children: Dict[str, List[str]] = {}
        all_n, child_n = set(), set()
        for e in edges:
            p, c = str(e.get("parent", "")).strip(), str(e.get("child", "")).strip()
            if p and c:
                children.setdefault(p, []).append(c)
                all_n.update([p, c])
                child_n.add(c)
        root = next(iter(all_n - child_n), None) or root_label

        def walk(n: str, prefix: str = "", depth: int = 0) -> List[str]:
            if depth > max_depth:
                return [prefix + "…"]
            kids = children.get(n, [])
            out: List[str] = []
            for i, k in enumerate(kids):
                last = i == len(kids) - 1
                out.append(prefix + ("└─ " if last else "├─ ") + k)
                out.extend(
                    walk(k, prefix + ("   " if last else "│  "), depth + 1)
                )
            return out

        return "\n".join([str(root)] + walk(str(root)))

    if isinstance(v, dict):
        text = "\n".join([root_label] + from_dict(v))
    elif isinstance(v, list) and (not v or isinstance(v[0], dict)):
        text = from_edges([e for e in v if isinstance(e, dict)])
    else:
        return "Invalid 'tree': expected a dict or list of {parent, child} edges."

    return _codeblock(title or "Tree", text)


# ── Ollama Tool Schema ───────────────────────────────────────────────────────


VISUALS_CORE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_visualization",
        "description": (
            "Generate ASCII/Markdown visualizations. "
            "Supports: table, comparison_table, chart, multi_chart, heatmap, timeline, flowchart, tree."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "viz_type": {
                    "type": "string",
                    "description": "Type of visualization: table | comparison_table | chart | multi_chart | heatmap | timeline | flowchart | tree",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title for the visualization",
                },
                "data": {
                    "type": "string",
                    "description": (
                        "JSON string of the data payload. For 'table': {rows:[{...}]}. "
                        "For 'chart': {x:[...], y:[...], chart_type:'bar|line|scatter'}. "
                        "For 'comparison_table': {items:[...], criteria:[...], scores:{...}}. "
                        "For 'heatmap': {data:[[...]], row_labels:[...], col_labels:[...]}. "
                        "For 'timeline': {events:[{date, label, detail}]}. "
                        "For 'flowchart': {steps:[{from, to, label}]}. "
                        "For 'tree': {tree:{...} or [{parent, child}]}."
                    ),
                },
            },
            "required": ["viz_type", "data"],
        },
    },
}

