export default function SentimentCards({ sentimentCounts, mixedCount, total }) {
  const cards = [
    { label: "Positive", count: sentimentCounts.positive, color: "#1D9E75" },
    { label: "Neutral",  count: sentimentCounts.neutral,  color: "#888780" },
    { label: "Negative", count: sentimentCounts.negative, color: "#E24B4A" },
    { label: "Mixed",    count: mixedCount,               color: "#534AB7" },
  ];

  return (
    <div className="stats-grid fade-in fade-in-delay-1">
      {cards.map(m => (
        <div key={m.label} className="stat-card">
          <div className="stat-value" style={{ color: m.color }}>{m.count}</div>
          <div className="stat-label">{m.label} · {total ? Math.round(m.count / total * 100) : 0}%</div>
        </div>
      ))}
    </div>
  );
}
