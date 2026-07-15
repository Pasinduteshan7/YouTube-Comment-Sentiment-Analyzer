import { useState, useEffect } from "react";
import axios from "axios";
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";

const API = "http://127.0.0.1:8000";

const SENT_COLORS = { positive: "#1D9E75", neutral: "#888780", negative: "#E24B4A" };

// Full 28-emotion color palette
const EMO_COLORS = {
  admiration:    "#7F77DD",
  amusement:     "#BA7517",
  anger:         "#E24B4A",
  annoyance:     "#D85A30",
  approval:      "#1D9E75",
  caring:        "#D4537E",
  confusion:     "#888780",
  curiosity:     "#378ADD",
  desire:        "#C0609A",
  disappointment:"#534AB7",
  disapproval:   "#A32D2D",
  disgust:       "#8B4513",
  embarrassment: "#CC7722",
  excitement:    "#E8A020",
  fear:          "#6B52A8",
  gratitude:     "#2E8B57",
  grief:         "#4A4A8A",
  joy:           "#BA7517",
  love:          "#D4537E",
  nervousness:   "#7B6B8A",
  optimism:      "#3A9E6A",
  pride:         "#9370DB",
  realization:   "#4682B4",
  relief:        "#5BA85B",
  remorse:       "#8B6969",
  sadness:       "#4169E1",
  surprise:      "#20B2AA",
  neutral:       "#888780",
};

const SUG_STYLE = {
  success: { bg: "#E1F5EE", color: "#085041", border: "#1D9E75" },
  warning: { bg: "#FAECE7", color: "#712B13", border: "#D85A30" },
  info:    { bg: "#E6F1FB", color: "#0C447C", border: "#378ADD" },
};

function formatNum(n) {
  if (!n) return "0";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function EmotionBadges({ emotions }) {
  // emotions can be an array (fresh analysis) or a comma-separated string (from CSV)
  const list = Array.isArray(emotions)
    ? emotions
    : typeof emotions === "string"
      ? emotions.split(",").map(e => e.trim()).filter(Boolean)
      : [];

  if (list.length === 0) return <span style={{ color: "#aaa", fontSize: 11 }}>—</span>;

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 3, justifyContent: "center" }}>
      {list.map((emo, i) => (
        <span key={i} style={{
          padding: "2px 6px", borderRadius: 6, fontSize: 10, fontWeight: 500,
          background: (EMO_COLORS[emo] || "#888") + "22",
          color: EMO_COLORS[emo] || "#666",
          border: `0.5px solid ${EMO_COLORS[emo] || "#888"}44`,
        }}>
          {emo}
        </span>
      ))}
    </div>
  );
}

function SentimentBadge({ c }) {
  if (c.is_mixed === true || c.is_mixed === "True") {
    return (
      <span
        title={`"${c.part1_text}" → ${c.part1_sentiment}\n"${c.part2_text}" → ${c.part2_sentiment}`}
        style={{ padding: "2px 8px", borderRadius: 8, fontSize: 11, fontWeight: 500,
          background: "#EDE8FB", color: "#3C3489", cursor: "help" }}
      >
        mixed
      </span>
    );
  }
  return (
    <span style={{
      padding: "2px 8px", borderRadius: 8, fontSize: 11, fontWeight: 500,
      background: c.sentiment === "positive" ? "#E1F5EE" : c.sentiment === "negative" ? "#FCEBEB" : "#F1EFE8",
      color:      c.sentiment === "positive" ? "#085041" : c.sentiment === "negative" ? "#A32D2D" : "#444",
    }}>
      {c.sentiment}
    </span>
  );
}

export default function App() {
  const [url, setUrl]               = useState("");
  const [max, setMax]               = useState(100);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const [data, setData]             = useState(null);
  const [filter, setFilter]         = useState("all");
  const [search, setSearch]         = useState("");
  const [topicOpen, setTopicOpen]   = useState(false);
  const [showAllEmo, setShowAllEmo] = useState(false);
  const [history, setHistory]       = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    axios.get(`${API}/last-analysis`)
      .then(res => { if (res.data.total > 0) setData(res.data); })
      .catch(() => {});
    axios.get(`${API}/history`)
      .then(res => setHistory(res.data.runs || []))
      .catch(() => {});
  }, []);

  async function analyse() {
    if (!url.trim()) return;
    setLoading(true); setError(""); setData(null); setFilter("all"); setSearch("");
    try {
      const res = await axios.post(`${API}/analyse`, { url: url.trim(), max_comments: max });
      setData(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || "Could not connect. Make sure backend is running: uvicorn api:app --reload");
    }
    setLoading(false);
  }

  const sentData = data ? [
    { name: "Positive", value: data.sentiment_counts?.positive || 0 },
    { name: "Neutral",  value: data.sentiment_counts?.neutral  || 0 },
    { name: "Negative", value: data.sentiment_counts?.negative || 0 },
  ] : [];

  const allEmoData = data
    ? Object.entries(data.emotion_counts || {})
        .map(([name, value]) => ({ name, value }))
        .filter(e => e.value > 0)
        .sort((a, b) => b.value - a.value)
    : [];

  const emoData = showAllEmo ? allEmoData : allEmoData.slice(0, 10);

  const isMixed = c => c.is_mixed === true || c.is_mixed === "True";

  const visible = data
    ? (data.comments || []).filter(c => {
        const mf = filter === "all"
          || (filter === "mixed" && isMixed(c))
          || (!isMixed(c) && c.sentiment === filter);
        const ms = (c.text || "").toLowerCase().includes(search.toLowerCase());
        return mf && ms;
      })
    : [];

  const vi         = data?.video_info || {};
  const total      = data?.total || 0;
  const mixedCount = data ? (data.comments || []).filter(isMixed).length : 0;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 980, margin: "0 auto", padding: "2rem 1rem", color: "#1a1a1a" }}>

      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 4 }}>YouTube Comment Sentiment Analyser</h1>
      <p style={{ color: "#888", fontSize: 14, marginBottom: 24 }}>
        {total > 0 ? `${total} comments analysed · 28-emotion fine-tuned model` : "Paste a YouTube URL below to get started"}
      </p>

      {/* URL input */}
      <div style={{ background: "#f9f9f7", border: "1px solid #e0e0e0", borderRadius: 12, padding: "1.25rem", marginBottom: 24 }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <input
            value={url} onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === "Enter" && analyse()}
            placeholder="https://www.youtube.com/watch?v=..."
            disabled={loading}
            style={{ flex: 1, minWidth: 260, padding: "9px 13px", borderRadius: 8, border: "1px solid #ddd", fontSize: 14 }}
          />
          <select value={max} onChange={e => setMax(Number(e.target.value))} disabled={loading}
            style={{ padding: "9px 12px", borderRadius: 8, border: "1px solid #ddd", fontSize: 14 }}>
            <option value={50}>50 comments</option>
            <option value={100}>100 comments</option>
            <option value={200}>200 comments</option>
            <option value={500}>500 comments</option>
          </select>
          <button onClick={analyse} disabled={loading} style={{
            padding: "9px 24px", borderRadius: 8, border: "none", fontSize: 14, fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
            background: loading ? "#ccc" : "#185FA5", color: "#fff",
          }}>
            {loading ? "Analysing…" : "Analyse"}
          </button>
          <button onClick={() => setShowHistory(o => !o)} style={{
            padding: "9px 16px", borderRadius: 8, border: "1px solid #ddd",
            fontSize: 13, cursor: "pointer", background: "transparent", color: "#555",
          }}>
            {showHistory ? "Hide history" : `History (${history.length})`}
          </button>
        </div>
        {error && <p style={{ color: "#E24B4A", fontSize: 13, marginTop: 10, marginBottom: 0 }}>{error}</p>}
      </div>

      {showHistory && history.length > 0 && (
        <div style={{ background: "#f9f9f7", border: "1px solid #e0e0e0", borderRadius: 12, padding: "1.25rem", marginBottom: 24 }}>
          <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 14 }}>Past analyses</p>
          <div style={{ display: "grid", gap: 8 }}>
            {history.map((run, i) => (
              <div key={run.run_id} style={{ background: "#fff", borderRadius: 8, padding: "10px 14px", border: "0.5px solid #e0e0e0", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "#333" }}>{run.video_title}</div>
                  <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>{run.timestamp} · {run.total} comments · {run.fingerprint}</div>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ fontSize: 11, color: "#1D9E75" }}>{run.positive_pct}% pos</span>
                  <span style={{ fontSize: 11, color: "#E24B4A" }}>{run.negative_pct}% neg</span>
                  <button onClick={() => { setUrl(run.url); setShowHistory(false); }}
                    style={{ fontSize: 11, padding: "3px 10px", borderRadius: 6, border: "0.5px solid #ddd", background: "none", cursor: "pointer", color: "#185FA5" }}>
                    Re-analyse
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: "center", padding: "4rem", color: "#888" }}>
          <div style={{ fontSize: 14, marginBottom: 6 }}>Fetching comments and running AI models…</div>
          <div style={{ fontSize: 12 }}>This takes about 20–40 seconds</div>
        </div>
      )}

      {data && !loading && (
        <>
          {/* Video info */}
          {vi.title && (
            <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1.25rem", marginBottom: 20, display: "flex", gap: 16, alignItems: "flex-start" }}>
              {vi.thumbnail && (
                <img src={vi.thumbnail} alt="thumbnail"
                  style={{ width: 160, height: 90, borderRadius: 8, objectFit: "cover", flexShrink: 0 }} />
              )}
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 5, lineHeight: 1.4 }}>{vi.title}</div>
                <div style={{ fontSize: 13, color: "#888", marginBottom: 12 }}>{vi.channel} · {vi.published}</div>
                <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                  {[
                    { label: "Views",    val: formatNum(vi.view_count) },
                    { label: "Likes",    val: formatNum(vi.like_count) },
                    { label: "Comments", val: formatNum(vi.comment_count) },
                    { label: "Analysed", val: total },
                  ].map(s => (
                    <div key={s.label} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: 16, fontWeight: 600 }}>{s.val}</div>
                      <div style={{ fontSize: 11, color: "#888" }}>{s.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Sentiment cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
            {[
              { label: "Positive", count: data.sentiment_counts.positive, color: "#1D9E75" },
              { label: "Neutral",  count: data.sentiment_counts.neutral,  color: "#888780" },
              { label: "Negative", count: data.sentiment_counts.negative, color: "#E24B4A" },
              { label: "Mixed",    count: mixedCount,                     color: "#534AB7" },
            ].map(m => (
              <div key={m.label} style={{ background: "#f5f5f3", borderRadius: 10, padding: "1rem", textAlign: "center" }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: m.color }}>{m.count}</div>
                <div style={{ fontSize: 12, color: "#888", marginTop: 4 }}>
                  {m.label} · {total ? Math.round(m.count / total * 100) : 0}%
                </div>
              </div>
            ))}
          </div>

          {/* Charts */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
            <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1rem" }}>
              <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 12 }}>Sentiment split</p>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={sentData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75}
                    label={({ name, percent }) => `${name} ${Math.round(percent * 100)}%`} labelLine={false}>
                    {sentData.map(e => <Cell key={e.name} fill={SENT_COLORS[e.name.toLowerCase()]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <p style={{ fontWeight: 500, fontSize: 14, margin: 0 }}>Emotion breakdown</p>
                {allEmoData.length > 10 && (
                  <button onClick={() => setShowAllEmo(o => !o)}
                    style={{ fontSize: 11, color: "#888", background: "none", border: "0.5px solid #ddd", borderRadius: 6, padding: "2px 8px", cursor: "pointer" }}>
                    {showAllEmo ? "show top 10" : `show all ${allEmoData.length}`}
                  </button>
                )}
              </div>
              <ResponsiveContainer width="100%" height={showAllEmo ? allEmoData.length * 24 : 260}>
                <BarChart data={emoData} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={80} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {emoData.map(e => <Cell key={e.name} fill={EMO_COLORS[e.name] || "#888"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Topic modelling */}
          {data.topics && data.topics.length > 0 && (
            <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1.25rem", marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onClick={() => setTopicOpen(o => !o)}>
                <p style={{ fontWeight: 500, fontSize: 14, margin: 0 }}>Topic breakdown</p>
                <span style={{ fontSize: 13, color: "#888" }}>{topicOpen ? "▲ hide" : "▼ show"}</span>
              </div>
              {topicOpen && (
                <div style={{ marginTop: 14, display: "grid", gap: 8 }}>
                  {data.topics.map((t, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ minWidth: 210, fontSize: 13, color: "#333" }}>{t.topic}</div>
                      <div style={{ flex: 1, background: "#f0f0ee", borderRadius: 4, height: 8, overflow: "hidden" }}>
                        <div style={{ width: `${t.percent}%`, background: "#378ADD", height: "100%", borderRadius: 4 }} />
                      </div>
                      <div style={{ fontSize: 12, color: "#888", minWidth: 70, textAlign: "right" }}>
                        {t.count} ({t.percent}%)
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Language breakdown */}
          {data.language_counts && Object.keys(data.language_counts).length > 1 && (
            <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1.25rem", marginBottom: 20 }}>
              <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 12 }}>Languages detected</p>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {Object.entries(data.language_counts).map(([lang, count]) => (
                  <div key={lang} style={{ background: "#f5f5f3", borderRadius: 8, padding: "6px 12px", textAlign: "center" }}>
                    <div style={{ fontSize: 14, fontWeight: 600 }}>{count}</div>
                    <div style={{ fontSize: 11, color: "#888" }}>{lang}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI Suggestions */}
          {data.suggestions && data.suggestions.length > 0 && (
            <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1.25rem", marginBottom: 20 }}>
              <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 14 }}>AI improvement suggestions</p>
              <div style={{ display: "grid", gap: 10 }}>
                {data.suggestions.map((s, i) => {
                  const c = SUG_STYLE[s.type] || SUG_STYLE.info;
                  return (
                    <div key={i} style={{ background: c.bg, border: `1px solid ${c.border}44`, borderLeft: `3px solid ${c.border}`, borderRadius: 8, padding: "10px 14px" }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: c.color, marginBottom: 3 }}>{s.title}</div>
                      <div style={{ fontSize: 12, color: "#555", lineHeight: 1.5 }}>{s.detail}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Pin these comments */}
          {data.pin_suggestions && (
            <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1.25rem", marginBottom: 20 }}>
              <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 4 }}>Comments worth replying to</p>
              <p style={{ fontSize: 12, color: "#888", marginBottom: 14 }}>These 3 comments will have the most impact if the creator replies to them</p>
              <div style={{ display: "grid", gap: 10 }}>
                {[
                  { key: "best_question",   label: "Top question",   color: "#378ADD", bg: "#E6F1FB" },
                  { key: "best_conflicted", label: "Most conflicted", color: "#534AB7", bg: "#EDE8FB" },
                  { key: "best_criticism",  label: "Top criticism",  color: "#E24B4A", bg: "#FCEBEB" },
                ].map(({ key, label, color, bg }) => {
                  const c = data.pin_suggestions[key];
                  if (!c) return null;
                  return (
                    <div key={key} style={{ background: bg, borderRadius: 8, padding: "10px 14px", borderLeft: `3px solid ${color}` }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color, marginBottom: 4 }}>{label} · {c.likes} likes</div>
                      <div style={{ fontSize: 13, color: "#333", lineHeight: 1.4 }}>{c.text}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Filters + search */}
          <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
            {["all", "positive", "neutral", "negative", "mixed"].map(f => (
              <button key={f} onClick={() => setFilter(f)} style={{
                padding: "5px 16px", borderRadius: 20, fontSize: 13, cursor: "pointer", fontWeight: 500,
                background: filter === f ? "#185FA5" : "transparent",
                color:      filter === f ? "#fff" : "#555",
                border:     filter === f ? "none" : "0.5px solid #ccc",
              }}>
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
            <input placeholder="Search comments…" value={search} onChange={e => setSearch(e.target.value)}
              style={{ marginLeft: "auto", padding: "6px 12px", borderRadius: 8, border: "0.5px solid #ccc", fontSize: 13, width: 200 }} />
          </div>

          {/* Comment table */}
          <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, overflow: "hidden" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 100px 160px 50px", background: "#f5f5f3", padding: "8px 14px", fontSize: 12, fontWeight: 500, color: "#666" }}>
              <span>Comment</span>
              <span style={{ textAlign: "center" }}>Sentiment</span>
              <span style={{ textAlign: "center" }}>Emotions</span>
              <span style={{ textAlign: "center" }}>Likes</span>
            </div>
            {visible.slice(0, 50).map((c, i) => (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 100px 160px 50px", padding: "10px 14px", borderTop: "0.5px solid #f0f0f0", fontSize: 13, alignItems: "center" }}>
                <span style={{ color: "#333", lineHeight: 1.4 }}>{c.text}</span>
                <span style={{ textAlign: "center" }}>
                  <SentimentBadge c={c} />
                </span>
                <span style={{ textAlign: "center" }}>
                  <EmotionBadges emotions={c.emotions} />
                </span>
                <span style={{ textAlign: "center", color: "#888", fontSize: 12 }}>{c.likes ?? "—"}</span>
              </div>
            ))}
            {visible.length === 0 && (
              <div style={{ padding: "2rem", textAlign: "center", color: "#888", fontSize: 14 }}>No comments match your filter.</div>
            )}
            {visible.length > 50 && (
              <div style={{ padding: "10px", textAlign: "center", color: "#888", fontSize: 12 }}>Showing 50 of {visible.length} comments</div>
            )}
          </div>
        </>
      )}
    </div>
  );
}