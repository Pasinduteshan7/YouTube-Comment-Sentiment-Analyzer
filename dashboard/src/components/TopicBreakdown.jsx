import { useState } from "react";

export default function TopicBreakdown({ topics }) {
  const [open, setOpen] = useState(false);

  if (!topics || topics.length === 0) return null;

  return (
    <div className="card fade-in fade-in-delay-3">
      <div className="topic-header" onClick={() => setOpen(o => !o)}>
        <span className="card-title" style={{ margin: 0 }}>Topic breakdown</span>
        <span className="topic-toggle">{open ? "▲ hide" : "▼ show"}</span>
      </div>
      {open && (
        <div style={{ marginTop: 14 }}>
          {topics.map((t, i) => (
            <div key={i} className="topic-row">
              <div className="topic-name">{t.topic}</div>
              <div className="topic-bar-track">
                <div className="topic-bar-fill" style={{ width: `${t.percent}%` }} />
              </div>
              <div className="topic-stat">{t.count} ({t.percent}%)</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
