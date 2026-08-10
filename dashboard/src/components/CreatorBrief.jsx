import ReactMarkdown from "react-markdown";

export default function CreatorBrief({ suggestions, pinSuggestions }) {
  return (
    <>
      {/* AI Strategic Brief */}
      {suggestions && (
        <div className="card ai-brief fade-in fade-in-delay-3">
          <div className="ai-brief-title">✨ Strategic Creator Brief</div>
          {typeof suggestions === "string" ? (
            <div className="ai-brief-content">
              <ReactMarkdown>{suggestions}</ReactMarkdown>
            </div>
          ) : (
            <div style={{ display: "grid", gap: 10 }}>
              {suggestions.map((s, i) => (
                <div key={i} className={`suggestion-card ${s.type || "info"}`}>
                  <div className="suggestion-title">{s.title}</div>
                  <div className="suggestion-detail">{s.detail}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Pin Suggestions */}
      {pinSuggestions && (
        <div className="card fade-in fade-in-delay-4">
          <div className="card-title">Comments worth replying to</div>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 14 }}>
            These 3 comments will have the most impact if the creator replies to them
          </p>
          <div style={{ display: "grid", gap: 10 }}>
            {[
              { key: "best_question",   label: "Top question",   color: "#378ADD", bg: "var(--accent-subtle)" },
              { key: "best_conflicted", label: "Most conflicted", color: "#534AB7", bg: "var(--mixed-bg)" },
              { key: "best_criticism",  label: "Top criticism",  color: "#E24B4A", bg: "var(--negative-bg)" },
            ].map(({ key, label, color, bg }) => {
              const c = pinSuggestions[key];
              if (!c) return null;
              return (
                <div key={key} className="pin-card" style={{ background: bg, borderColor: color }}>
                  <div className="pin-label" style={{ color }}>{label} · {c.likes} likes</div>
                  <div className="pin-text">{c.text}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}
