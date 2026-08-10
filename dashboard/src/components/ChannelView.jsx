import { LineChart, Line, CartesianGrid, Legend, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { EmotionBadges, SentimentBadge } from "./Badges";
import { EMO_COLORS } from "../constants";

export default function ChannelView({ data, setUrl, setActiveTab }) {
  if (!data) return null;

  return (
    <div className="fade-in">
      {/* Summary Cards */}
      <div className="stats-grid">
        {[
          { label: "Videos analysed", val: data.total_videos, color: "var(--accent)" },
          { label: "Avg positive",    val: data.avg_positive_pct + "%", color: "#1D9E75" },
          { label: "Avg negative",    val: data.avg_negative_pct + "%", color: "#E24B4A" },
          { label: "Common profile",  val: data.most_common_profile, color: "#534AB7" },
        ].map(m => (
          <div key={m.label} className="stat-card">
            <div className="stat-value" style={{ color: m.color, fontSize: m.label === "Common profile" ? "1rem" : undefined }}>{m.val}</div>
            <div className="stat-label">{m.label}</div>
          </div>
        ))}
      </div>

      {/* Highlights */}
      <div className="highlight-grid">
        <div className="highlight-card best">
          <div className="highlight-label" style={{ color: "#1D9E75" }}>Best received video</div>
          <div className="highlight-value" style={{ color: "var(--positive-text)" }}>{data.best_received_video}</div>
        </div>
        <div className="highlight-card worst">
          <div className="highlight-label" style={{ color: "#E24B4A" }}>Most divisive video</div>
          <div className="highlight-value" style={{ color: "var(--negative-text)" }}>{data.most_divisive_video}</div>
        </div>
      </div>

      {/* Trend Chart */}
      <div className="card">
        <div className="card-title">Sentiment Trend Over Time</div>
        <div style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={[...data.videos].reverse()} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-secondary)" />
              <XAxis dataKey="title" tick={{ fontSize: 10, fill: "var(--text-muted)" }}
                tickFormatter={val => val.length > 20 ? val.substring(0,20)+"..." : val} />
              <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} domain={[0, 100]} unit="%" />
              <Tooltip contentStyle={{ borderRadius: 8, fontSize: 12, border: "1px solid var(--border-primary)", background: "var(--bg-secondary)", color: "var(--text-primary)" }} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
              <Line type="monotone" dataKey="positive_pct" name="Positive %" stroke="#1D9E75" strokeWidth={3} activeDot={{ r: 6 }} />
              <Line type="monotone" dataKey="negative_pct" name="Negative %" stroke="#E24B4A" strokeWidth={3} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-video Cards */}
      <div style={{ display: "grid", gap: 12 }}>
        {data.videos.map((v, i) => (
          <div key={i} className="card video-card">
            {v.thumbnail && <img src={v.thumbnail} alt="" className="video-thumb" />}
            <div className="video-info">
              <div className="video-title">{v.title}</div>
              <div className="video-meta">{v.published} · {v.total} comments analysed</div>
              <div className="video-badges">
                <span className="mini-badge" style={{ background: "var(--positive-bg)", color: "var(--positive-text)" }}>{v.positive_pct}% positive</span>
                <span className="mini-badge" style={{ background: "var(--negative-bg)", color: "var(--negative-text)" }}>{v.negative_pct}% negative</span>
                <span className="mini-badge" style={{ background: "var(--mixed-bg)", color: "var(--mixed-text)" }}>{v.fingerprint.profile}</span>
                {v.top_emotions.slice(0, 3).map(([emo]) => (
                  <span key={emo} className="emo-badge" style={{
                    background: (EMO_COLORS[emo] || "#888") + "22",
                    color: EMO_COLORS[emo] || "#666",
                    border: `0.5px solid ${EMO_COLORS[emo] || "#888"}44`,
                  }}>{emo}</span>
                ))}
                <button className="btn-link" style={{ marginLeft: "auto" }}
                  onClick={() => { setUrl(v.url); setActiveTab("single"); }}>
                  Full analysis
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Latest Comments */}
      {data.latest_comments && data.latest_comments.length > 0 && (
        <div className="comment-table" style={{ marginTop: 24 }}>
          <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border-secondary)", fontWeight: 600, fontSize: "0.88rem" }}>
            Latest Comments Across Channel
          </div>
          <div className="comment-header">
            <span>Comment</span>
            <span style={{ textAlign: "center" }}>Sentiment</span>
            <span style={{ textAlign: "center" }}>Emotions</span>
            <span style={{ textAlign: "center" }}>Likes</span>
          </div>
          {data.latest_comments.map((c, i) => (
            <div key={i} className="comment-row">
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span className="comment-text">{c.text}</span>
                <span style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>from {c.video_title}</span>
              </div>
              <span style={{ textAlign: "center" }}><SentimentBadge c={c} /></span>
              <span style={{ textAlign: "center" }}><EmotionBadges emotions={c.emotions} /></span>
              <span className="comment-likes">{c.likes ?? "—"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
