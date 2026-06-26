"use strict";
// Usage: node slides_generator.js <data_json_path> <output_pptx_path>

const pptxgen = require("pptxgenjs");
const fs = require("fs");

const dataPath = process.argv[2];
const outPath  = process.argv[3];

if (!dataPath || !outPath) {
  console.error("Usage: node slides_generator.js <data.json> <output.pptx>");
  process.exit(1);
}

const { scan, hypotheses: rawHyps } = JSON.parse(fs.readFileSync(dataPath, "utf8"));

const company = scan.company_name || "Okänt bolag";

const hyps = [...(rawHyps || [])]
  .filter(Boolean)
  .sort((a, b) => (a.simulation_priority_rank || 99) - (b.simulation_priority_rank || 99));

const CLUSTER_COL = {
  demand_volatility:         "Volym & prognos",
  price_volatility:          "Pris & marknad",
  capacity_variability:      "Kapacitet & drift",
  flow_fragmentation:        "Kvalitet & flöde",
  knowledge_dependency:      "Kunskap & process",
  margin_pressure:           "Pris & lönsamhet",
  risk_uncertainty:          "Risk & osäkerhet",
  portfolio_complexity:      "Portfölj & mix",
  project_uncertainty:       "Projekt & genomförande",
  working_capital_imbalance: "Kapital & likviditet",
  unknown:                   "Övrigt",
};

function colTitle(h) {
  return CLUSTER_COL[h.root_cause_cluster] || (h.title || "").slice(0, 24);
}

function quantText(h) {
  const qt   = h.quantification_targets || [];
  const parts = [];
  for (const t of qt.slice(0, 2)) {
    const m = t.metric || "", q = t.question || "";
    if (m && q) parts.push(`${m}: ${q}`);
    else if (q) parts.push(q);
    else if (m) parts.push(m);
  }
  const dr    = h.data_requirements || [];
  const names = dr.slice(0, 3).map(d => d.data_name || "").filter(Boolean).join(", ");
  if (names) parts.push(`Databehov: ${names}`);
  return parts.join(" | ");
}

function discussionQ(h) {
  const vqs = h.validation_questions || [];
  return vqs[0] || (h.decision || {}).decision_description || "";
}

// ─── Constants ───────────────────────────────────────────────────────────────
const NAVY   = "1A2744";
const WHITE  = "FFFFFF";
const BG     = "EDF1F6";
const BORDER = "C8CDD6";

const ML = 0.5;              // horizontal margin
const UW = 13.33 - 2 * ML;  // usable width ≈ 12.33"
const SH = 7.5;              // slide height

// ─── Presentation setup ──────────────────────────────────────────────────────
const pres = new pptxgen();
pres.layout  = "LAYOUT_WIDE";  // 13.33" × 7.5"
pres.title   = `Discovery Workshop – ${company}`;
pres.author  = "xZero";

function newSlide() {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

function addTitle(slide, text, y = 0.3) {
  slide.addText(text, {
    x: ML, y, w: UW, h: 0.85,
    fontSize: 26, bold: true, color: NAVY,
    align: "center", valign: "middle", margin: 0,
  });
}

// ─── Slide 1: Framing ────────────────────────────────────────────────────────
{
  const slide = newSlide();

  slide.addText(
    "“Om ni kunde förbättra EN sak som direkt påverkar resultatet, vad skulle det vara?”",
    {
      x: ML, y: 0.55, w: UW, h: 1.0,
      fontSize: 22, italic: true, color: NAVY,
      align: "center", valign: "middle", margin: 0,
    }
  );

  const n       = Math.min(hyps.length, 3);
  const gap     = 0.22;
  const colW    = (UW - gap * (n - 1)) / n;
  const headerH = 0.55;
  const bodyH   = 4.15;
  const cardTop = 1.75;

  hyps.slice(0, 3).forEach((h, i) => {
    const colX = ML + i * (colW + gap);

    // Navy header rectangle
    slide.addShape(pres.shapes.RECTANGLE, {
      x: colX, y: cardTop, w: colW, h: headerH,
      fill: { color: NAVY }, line: { color: NAVY, width: 0 },
    });
    slide.addText(colTitle(h), {
      x: colX, y: cardTop, w: colW, h: headerH,
      fontSize: 14, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });

    // White body rectangle
    slide.addShape(pres.shapes.RECTANGLE, {
      x: colX, y: cardTop + headerH, w: colW, h: bodyH,
      fill: { color: WHITE }, line: { color: BORDER, width: 0.5 },
    });

    const syms = (h.symptoms || []).slice(0, 3);
    if (syms.length) {
      const items = syms.map((s, idx) => ({
        text: `→  ${s}`,
        options: { breakLine: idx < syms.length - 1, fontSize: 12, paraSpaceAfter: 10 },
      }));
      slide.addText(items, {
        x: colX + 0.18, y: cardTop + headerH + 0.25,
        w: colW - 0.36, h: bodyH - 0.35,
        color: NAVY, valign: "top", margin: 0,
      });
    }
  });
}

// ─── Slide 2: Hypoteser overview ─────────────────────────────────────────────
{
  const slide = newSlide();
  addTitle(slide, "Hypoteser");

  const rowH = Math.min(1.1, (SH - 1.6) / hyps.length);
  const rows = hyps.map(h => {
    const hnum = (h.hypothesis_id || "").replace(/^H/, "") || "?";
    return [
      {
        text: `Hypotes ${hnum}`,
        options: { bold: true, color: NAVY, fontSize: 14, align: "left", valign: "middle" },
      },
      {
        text: h.title || "",
        options: { color: NAVY, fontSize: 14, align: "left", valign: "middle" },
      },
    ];
  });

  slide.addTable(rows, {
    x: ML, y: 1.45, w: UW,
    rowH, colW: [2.2, UW - 2.2],
    fill: { color: WHITE },
    border: { pt: 0.5, color: BORDER },
    fontFace: "Calibri",
  });
}

// ─── Slides 3-N: Simple hypothesis slides ────────────────────────────────────
hyps.forEach(h => {
  const slide = newSlide();
  const hnum  = (h.hypothesis_id || "").replace(/^H/, "");
  addTitle(slide, `Hypotes ${hnum}`);

  const mech = (h.mechanism || {}).description || "";

  // White card
  slide.addShape(pres.shapes.RECTANGLE, {
    x: ML, y: 1.35, w: UW, h: 2.2,
    fill: { color: WHITE }, line: { color: BORDER, width: 0.5 },
  });
  slide.addText(
    [
      { text: `Hypotes ${hnum}  ${h.title || ""}`, options: { bold: true, breakLine: true, fontSize: 13, paraSpaceAfter: 5 } },
      { text: mech, options: { fontSize: 11.5 } },
    ],
    {
      x: ML + 0.22, y: 1.45, w: UW - 0.44, h: 2.0,
      color: NAVY, valign: "top", margin: 0,
    }
  );

  // Three discussion questions
  const qs = [
    "Känner ni igen det här problemet?",
    "Hur stort är problemet hos er idag?",
    "Har ni data för att gå vidare?",
  ];
  const qItems = qs.map((q, i) => ({
    text: `→  ${q}`,
    options: { breakLine: i < 2, fontSize: 15, bold: false, paraSpaceAfter: 10 },
  }));
  slide.addText(qItems, {
    x: ML + 1.2, y: 3.75, w: UW - 2.4, h: 3.1,
    color: NAVY, valign: "top", margin: 0,
  });
});

// ─── Slide: Prioritering ─────────────────────────────────────────────────────
{
  const slide = newSlide();
  addTitle(slide, "Prioritering");

  const items = hyps.map((h, i) => ({
    text: `${i + 1}.  ${h.hypothesis_id || ("H" + (i + 1))} – ${h.title || ""}`,
    options: { breakLine: i < hyps.length - 1, fontSize: 18, bold: true, paraSpaceAfter: 14 },
  }));

  slide.addText(items, {
    x: ML + 1.5, y: 1.5, w: UW - 3.0, h: 5.2,
    color: NAVY, valign: "top", margin: 0,
  });
}

// ─── Slides: Detailed hypothesis slides ──────────────────────────────────────
hyps.forEach(h => {
  const slide = newSlide();
  const hnum  = (h.hypothesis_id || "").replace(/^H/, "");
  const ct    = colTitle(h);
  addTitle(slide, `Hypotes ${hnum}${ct ? " – " + ct : ""}`);

  const dec    = h.decision || {};
  const beslut = dec.decision_name || dec.decision_description || "";
  const symp   = (h.symptoms || []).join(" • ");

  const labelOpts = { bold: true, color: NAVY, fontSize: 11, valign: "top", align: "left" };
  const valueOpts = (bold = false) => ({ bold, color: NAVY, fontSize: 11, valign: "top", align: "left" });

  const rows = [
    [
      { text: "Hypotes",      options: labelOpts },
      { text: h.title || "", options: valueOpts(true) },
    ],
    [
      { text: "Mekanism",     options: labelOpts },
      { text: (h.mechanism || {}).description || "–", options: valueOpts() },
    ],
    [
      { text: "Beslut",       options: labelOpts },
      { text: beslut || "–",  options: valueOpts() },
    ],
    [
      { text: "Symptom",      options: labelOpts },
      { text: symp || "–",   options: valueOpts() },
    ],
    [
      { text: "Kvantifiering", options: labelOpts },
      { text: quantText(h) || "–", options: valueOpts() },
    ],
    [
      { text: "Diskussion",   options: labelOpts },
      { text: discussionQ(h) || "–", options: valueOpts() },
    ],
  ];

  slide.addTable(rows, {
    x: ML, y: 1.38, w: UW, h: 5.7,
    colW: [1.55, UW - 1.55],
    fill: { color: WHITE },
    border: { pt: 0.5, color: BORDER },
    fontFace: "Calibri",
  });
});

// ─── Write output ─────────────────────────────────────────────────────────────
pres.writeFile({ fileName: outPath })
  .then(() => process.exit(0))
  .catch(err => { console.error(err); process.exit(1); });
