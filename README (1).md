# SVI Dashboard — AI Automation & Entry-Level Jobs

An interactive data-visualization dashboard built from the findings of the research paper
**"The Economic Impact of AI Automation on Entry-Level Jobs"** by Sanchita Sharma,
Department of Computer Science and Engineering, Chitkara University, Punjab, India.

---

## What This Project Does

The dashboard turns the paper's static tables and figures into an interactive, browser-based
experience. It covers four main areas:

| Section | What it shows |
|---|---|
| **At a Glance** | Six headline statistics from the paper |
| **SVI Chart** | Animated bar chart with sector filter controls; hover tooltip |
| **Demographic Vulnerability** | Scroll-animated bars broken down by education, race/ethnicity, gender and age |
| **Model Validation** | R², RMSE and methodology summary |
| **Policy Pillars** | Four-pillar Multi-Stakeholder Policy Framework |

---

## File Structure

```
project/
├── svi_dashboard.html   ← Single-file application (HTML + CSS + JS)
└── README.md            ← This file
```

The entire project ships as **one self-contained HTML file**. There is no build step,
no package manager, and no server required.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| A modern web browser | Any — 2020 or newer | Chrome, Firefox, Edge, Safari all work |
| Internet connection | Optional | Only needed to load Google Fonts (Playfair Display, DM Mono, DM Sans). Falls back to serif/monospace if offline. |

No Node.js, Python, or any other runtime is needed.

---

## How to Run

### Option A — Double-click (simplest)

1. Download or copy `svi_dashboard.html` to any folder on your machine.
2. Double-click the file. Your default browser opens the dashboard immediately.

### Option B — Local development server (recommended for editing)

Using Python (comes pre-installed on macOS and most Linux distros):

```bash
# Python 3
python3 -m http.server 8080

# Python 2 (legacy)
python -m SimpleHTTPServer 8080
```

Then open `http://localhost:8080/svi_dashboard.html` in your browser.

Using Node.js (if installed):

```bash
npx serve .
# or
npx http-server -p 8080
```

### Option C — VS Code Live Server

1. Install the **Live Server** extension (Ritwick Dey) from the VS Code marketplace.
2. Right-click `svi_dashboard.html` in the Explorer panel.
3. Choose **"Open with Live Server"**.

---

## How to Interact With the Dashboard

- **Filter buttons** above the SVI bar chart let you narrow sectors by risk tier
  (`All`, `Critical >80`, `High >65`, `Below Threshold`).
- **Hover** over any bar row to see a tooltip with the sector's SVI score, job-loss
  percentage and risk classification.
- **Scroll** down the page — every section animates in on first entry.
- **Resize** the browser window — the layout is fully responsive.

---

## Data Sources

All data points in this dashboard are drawn directly from the paper cited below.
The paper's own sources include:

- U.S. Bureau of Labor Statistics (BLS) Occupational Employment and Wage Statistics
- World Economic Forum *Future of Jobs Report* 2023
- Stanford Human-Centered AI Institute *AI Index Report* 2023
- McKinsey Global Institute (2023)
- Occupational Information Network (O\*NET)
- 47 peer-reviewed studies (2018–2024) reviewed under PRISMA guidelines

---

## Customisation Guide

All data lives in three clearly labelled JavaScript arrays near the top of the `<script>`
block. You can edit them without touching any rendering logic.

```js
// ── 1. Sector SVI scores ──────────────────────────────────────
const SECTORS = [
  { name: 'Financial Services', svi: 82, jobLoss: 9.7, risk: 'Critical', ... },
  // add or remove objects here
];

// ── 2. Demographic breakdown groups ──────────────────────────
const DEMO_DATA = [
  { title: 'By Educational Attainment', rows: [...], note: '...' },
  // add or remove groups here
];

// ── 3. Policy pillar descriptions ────────────────────────────
const PILLARS = [
  { num: 'I', tag: 'Pillar One', title: '...', body: '...' },
  // add or remove pillars here
];
```

### Changing the risk threshold line

The golden threshold line on the SVI chart is positioned at 65% of the bar track width
(matching SVI = 65 on a 0–100 scale). To change it, find this CSS rule:

```css
.threshold-line { left: calc(65% - 1px); }
.threshold-label { left: calc(65% - 24px); }
```

Replace `65%` with your target value (e.g. `70%` for SVI = 70).

### Changing colours

All colours are CSS custom properties at the top of the `<style>` block:

```css
:root {
  --accent:   #e8c84a;   /* gold highlight */
  --critical: #e05a4e;   /* red — critical risk */
  --high:     #e8c84a;   /* gold — high risk */
  --moderate: #82c96e;   /* green — moderate */
  --low:      #4ec9a4;   /* teal — low risk */
  /* ... */
}
```

---

## Code Architecture

The JavaScript is split into clearly named functions with JSDoc comments:

| Function | Purpose |
|---|---|
| `renderBars(filter)` | Builds the animated SVI bar chart; re-runs on filter change |
| `renderTable()` | Populates the data table from `SECTORS` |
| `renderDemographics()` | Creates demographic exposure cards |
| `renderPillars()` | Inserts policy pillar cards |
| `showTooltip(e, sector)` | Displays hover tooltip |
| `moveTooltip(e)` | Keeps tooltip tracking the cursor |
| `hideTooltip()` | Hides tooltip on mouse-leave |
| `initObserver()` | IntersectionObserver for scroll-triggered animations |
| `initFilters()` | Wires filter button click events |
| `init()` | Entry point — calls all of the above |

---

## Citation

> Sharma, S. (2024). *The Economic Impact of AI Automation on Entry-Level Jobs.*
> Department of Computer Science and Engineering, Chitkara University, Rajpura, Punjab, India.

---

## License

This visualization code is released for educational and non-commercial use.
The underlying research data belongs to the original paper's author and cited sources.
