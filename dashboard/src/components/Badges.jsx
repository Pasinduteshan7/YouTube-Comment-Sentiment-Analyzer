import { EMO_COLORS } from "../constants";

export function EmotionBadges({ emotions }) {
  const list = Array.isArray(emotions)
    ? emotions
    : typeof emotions === "string"
      ? emotions.split(",").map(e => e.trim()).filter(Boolean)
      : [];
  if (list.length === 0) return <span style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>—</span>;
  return (
    <div className="emotion-badges">
      {list.map((emo, i) => (
        <span key={i} className="emo-badge" style={{
          background: (EMO_COLORS[emo] || "#888") + "22",
          color: EMO_COLORS[emo] || "#666",
          border: `0.5px solid ${EMO_COLORS[emo] || "#888"}44`,
        }}>{emo}</span>
      ))}
    </div>
  );
}

export function SentimentBadge({ c }) {
  if (c.is_mixed === true || c.is_mixed === "True") {
    return (
      <span className="sentiment-badge mixed"
        title={`"${c.part1_text}" → ${c.part1_sentiment}\n"${c.part2_text}" → ${c.part2_sentiment}`}>
        mixed
      </span>
    );
  }
  const cls = c.sentiment === "positive" ? "positive" : c.sentiment === "negative" ? "negative" : "neutral";
  return <span className={`sentiment-badge ${cls}`}>{c.sentiment}</span>;
}
