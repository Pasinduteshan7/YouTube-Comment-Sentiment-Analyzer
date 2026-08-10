/** Shared constants and utility functions used across components. */

export const API = "http://127.0.0.1:8000";

export const SENT_COLORS = { positive: "#1D9E75", neutral: "#888780", negative: "#E24B4A" };

export const EMO_COLORS = {
  admiration: "#7F77DD", amusement: "#BA7517", anger: "#E24B4A", annoyance: "#D85A30",
  approval: "#1D9E75", caring: "#D4537E", confusion: "#888780", curiosity: "#378ADD",
  desire: "#C0609A", disappointment: "#534AB7", disapproval: "#A32D2D", disgust: "#8B4513",
  embarrassment: "#CC7722", excitement: "#E8A020", fear: "#6B52A8", gratitude: "#2E8B57",
  grief: "#4A4A8A", joy: "#BA7517", love: "#D4537E", nervousness: "#7B6B8A",
  optimism: "#3A9E6A", pride: "#9370DB", realization: "#4682B4", relief: "#5BA85B",
  remorse: "#8B6969", sadness: "#4169E1", surprise: "#20B2AA", neutral: "#888780",
};

export function formatNum(n) {
  if (!n) return "0";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

export function isMixed(c) {
  return c.is_mixed === true || c.is_mixed === "True";
}

export function exportCSV(comments) {
  if (!comments || comments.length === 0) return;
  const headers = ["text", "sentiment", "sentiment_score", "emotions", "likes", "is_mixed", "language"];
  const rows = comments.map(c => headers.map(h => {
    const val = c[h];
    if (Array.isArray(val)) return val.join("; ");
    const str = String(val ?? "");
    return str.includes(",") || str.includes('"') ? `"${str.replace(/"/g, '""')}"` : str;
  }));
  const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sentiment_analysis_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportJSON(data) {
  if (!data) return;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sentiment_analysis_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
