---
name: visuals_core
description: Generate ASCII/Markdown visualizations — tables, charts, timelines, flowcharts, trees, and heatmaps. Discord-safe, no HTML or CDN dependencies.
metadata: {"clawhub":{"emoji":"📊","requires":{},"tools":["visuals_core"]}}
---

# visuals_core

Use `visuals_core` to present structured data visually inside Discord.
All output is plain ASCII/Markdown — no images, no embeds, no external dependencies.

## Tool signature

```
visuals_core(viz_type: str, data: str, title: str = "") -> str
```

- `viz_type` — one of the types listed below
- `data` — a JSON string containing the data payload (see each type for shape)
- `title` — optional heading shown above the visualization

**Important:** `data` must be a JSON-encoded string, not a raw dict.

## Visualization types

### table
Render a list of dicts as a Markdown table.
```json
{"rows": [{"Name": "Alice", "Score": 95}, {"Name": "Bob", "Score": 88}]}
```

### comparison_table
Compare items across criteria, with optional best-value highlighting (★).
```json
{
  "items": ["iPhone", "Pixel"],
  "criteria": ["Price", "Battery", "Camera"],
  "scores": {
    "iPhone":  {"Price": 999, "Battery": 8, "Camera": 9},
    "Pixel":   {"Price": 799, "Battery": 9, "Camera": 10}
  }
}
```

### chart
Single-series bar, line, or scatter chart.
```json
{"x": ["Jan","Feb","Mar"], "y": [10, 25, 18], "chart_type": "bar"}
```
`chart_type` options: `bar` | `line` | `scatter` (default: `line`)

### multi_chart
Multiple series on separate ASCII charts.
```json
{
  "series": [
    {"name": "Revenue", "x": ["Q1","Q2","Q3"], "y": [100, 130, 120]},
    {"name": "Costs",   "x": ["Q1","Q2","Q3"], "y": [80,  90,  95]}
  ],
  "chart_type": "line"
}
```

### heatmap
2D grid with density shading (░▒▓█).
```json
{
  "data": [[1,2,3],[4,5,6],[7,8,9]],
  "row_labels": ["Mon","Tue","Wed"],
  "col_labels": ["Morning","Afternoon","Evening"]
}
```

### timeline
Chronological event list.
```json
{
  "events": [
    {"date": "2024-01", "label": "Project kick-off", "detail": "Team assembled"},
    {"date": "2024-03", "label": "Beta launch",       "detail": "50 users onboarded"}
  ]
}
```

### flowchart
Directed graph of steps. `direction`: `tb` (top-to-bottom) or `lr` (left-to-right).
```json
{
  "steps": [
    {"from": "Start", "to": "Process", "label": "begin"},
    {"from": "Process", "to": "End",   "label": "done"}
  ],
  "direction": "tb"
}
```

### tree
Hierarchical structure from a nested dict or parent/child edge list.

Dict form:
```json
{"tree": {"Root": {"Branch A": ["Leaf 1", "Leaf 2"], "Branch B": ["Leaf 3"]}}}
```

Edge list form:
```json
{"tree": [{"parent": "Root", "child": "A"}, {"parent": "Root", "child": "B"}]}
```

## Full call example

```python
visuals_core(
    viz_type="chart",
    title="Monthly Sales",
    data='{"x":["Jan","Feb","Mar"],"y":[42,67,55],"chart_type":"bar"}'
)
```

## Notes

- Always serialize `data` as a JSON string — not a Python dict
- Max 250 rows for tables/timelines, 60 columns for tables
- Chart height is fixed at 12 rows; keep x-axis labels short (≤3 chars render best)
- Output is appended to the bot reply as a Discord code block — no extra formatting needed
- Use `comparison_table` when ranking or comparing options for the user
