import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

import { API, isMixed, exportCSV, exportJSON } from "./constants";
import LoadingSkeleton from "./components/LoadingSkeleton";
import VideoInfo from "./components/VideoInfo";
import SentimentCards from "./components/SentimentCards";
import Charts from "./components/Charts";
import TopicBreakdown from "./components/TopicBreakdown";
import CreatorBrief from "./components/CreatorBrief";
import CommentTable from "./components/CommentTable";
import ChannelView from "./components/ChannelView";
import TimelineChart from "./components/TimelineChart";

export default function App() {
  const [url, setUrl] = useState("");
  const [max, setMax] = useState(100);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  
  const [channelUrl, setChannelUrl] = useState("");
  const [maxVideos, setMaxVideos] = useState(5);
  const [channelLoading, setChannelLoading] = useState(false);
  const [channelData, setChannelData] = useState(null);
  const [channelError, setChannelError] = useState("");
  
  const [activeTab, setActiveTab] = useState("single");
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

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
      // Refresh history
      axios.get(`${API}/history`).then(r => setHistory(r.data.runs || [])).catch(() => {});
    } catch (e) {
      setError(e?.response?.data?.detail || "Could not connect. Make sure backend is running.");
    }
    setLoading(false);
  }

  async function analyseChannel() {
    if (!channelUrl.trim()) return;
    setChannelLoading(true); setChannelError(""); setChannelData(null);
    try {
      const res = await axios.post(`${API}/analyse-channel`, {
        url: channelUrl.trim(), max_videos: maxVideos, comments_per_video: 100,
      });
      setChannelData(res.data);
    } catch (e) {
      setChannelError(e?.response?.data?.detail || "Could not connect to backend.");
    }
    setChannelLoading(false);
  }

  // Pre-process data for charts
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

  const visibleComments = data
    ? (data.comments || []).filter(c => {
        const mf = filter === "all"
          || (filter === "mixed" && isMixed(c))
          || (filter === "toxic" && c.is_toxic)
          || (!isMixed(c) && c.sentiment === filter && filter !== "toxic");
        const ms = (c.text || "").toLowerCase().includes(search.toLowerCase());
        return mf && ms;
      })
    : [];

  const total = data?.total || 0;
  const mixedCount = data ? (data.comments || []).filter(isMixed).length : 0;

  return (
    <div className="app-container">

      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="app-header">
        <div>
          <h1>YouTube Comment Sentiment Analyser</h1>
          <p className="subtitle">
            {total > 0 ? `${total} comments analysed · 28-emotion fine-tuned model` : "Paste a YouTube URL below to get started"}
          </p>
        </div>
        <button className="theme-toggle" onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "☀️" : "🌙"} {theme === "dark" ? "Light" : "Dark"}
        </button>
      </header>

      {/* ── Tabs ────────────────────────────────────────────────── */}
      <div className="tab-bar">
        {["single", "channel"].map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`tab-btn ${activeTab === tab ? "active" : ""}`}>
            {tab === "single" ? "Single video" : "Channel / playlist"}
          </button>
        ))}
      </div>

      {/* ── Single Video Input ──────────────────────────────────── */}
      {activeTab === "single" && (
        <div className="card input-card fade-in">
          <div className="input-row">
            <input className="url-input" value={url} onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === "Enter" && analyse()}
              placeholder="https://www.youtube.com/watch?v=..." disabled={loading} />
            <select className="select-input" value={max} onChange={e => setMax(Number(e.target.value))} disabled={loading}>
              <option value={50}>50 comments</option>
              <option value={100}>100 comments</option>
              <option value={200}>200 comments</option>
              <option value={500}>500 comments</option>
            </select>
            <button className="btn-primary" onClick={analyse} disabled={loading}>
              {loading ? "Analysing…" : "Analyse"}
            </button>
            <button className="btn-secondary" onClick={() => setShowHistory(o => !o)}>
              {showHistory ? "Hide history" : `History (${history.length})`}
            </button>
          </div>
          {error && <p className="error-text">{error}</p>}
        </div>
      )}

      {/* ── Channel Input ───────────────────────────────────────── */}
      {activeTab === "channel" && (
        <div className="card input-card fade-in">
          <div className="input-row">
            <input className="url-input" value={channelUrl} onChange={e => setChannelUrl(e.target.value)}
              onKeyDown={e => e.key === "Enter" && analyseChannel()}
              placeholder="https://www.youtube.com/@ChannelName or playlist URL" disabled={channelLoading} />
            <select className="select-input" value={maxVideos} onChange={e => setMaxVideos(Number(e.target.value))} disabled={channelLoading}>
              <option value={3}>Last 3 videos</option>
              <option value={5}>Last 5 videos</option>
              <option value={10}>Last 10 videos</option>
            </select>
            <button className="btn-primary" onClick={analyseChannel} disabled={channelLoading}>
              {channelLoading ? "Scanning…" : "Scan channel"}
            </button>
          </div>
          {channelError && <p className="error-text">{channelError}</p>}
          <p className="hint-text">Scans the last N videos. Each video takes ~20 seconds — allow 2-3 minutes for 5 videos.</p>
        </div>
      )}

      {/* ── History Panel ───────────────────────────────────────── */}
      {showHistory && history.length > 0 && activeTab === "single" && (
        <div className="card fade-in">
          <div className="card-title">Past analyses</div>
          <div style={{ display: "grid", gap: 8 }}>
            {history.map(run => (
              <div key={run.run_id} className="history-item">
                <div>
                  <div className="history-title">{run.video_title}</div>
                  <div className="history-meta">{run.timestamp} · {run.total} comments · {run.fingerprint}</div>
                </div>
                <div className="history-stats">
                  <span className="history-pct pos">{run.positive_pct}% pos</span>
                  <span className="history-pct neg">{run.negative_pct}% neg</span>
                  <button className="btn-link" onClick={() => { setUrl(run.url); setShowHistory(false); }}>
                    Re-analyse
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Loading ─────────────────────────────────────────────── */}
      {activeTab === "single" && loading && <LoadingSkeleton type="single" />}
      {activeTab === "channel" && channelLoading && <LoadingSkeleton type="channel" />}

      {/* ── Single Video Results ────────────────────────────────── */}
      {activeTab === "single" && data && !loading && (
        <>
          <VideoInfo vi={data.video_info || {}} total={total} />
          
          <SentimentCards 
            sentimentCounts={data.sentiment_counts} 
            mixedCount={mixedCount} 
            toxicCount={data.toxic_count}
            total={total} 
          />

          {/* Emotional Fingerprint */}
          {data.fingerprint && data.fingerprint.profile && (
            <div className="card fingerprint-card fade-in fade-in-delay-1">
              <div className="fingerprint-profile">🧬 {data.fingerprint.profile}</div>
              <div className="fingerprint-desc">{data.fingerprint.description}</div>
            </div>
          )}

          <Charts sentData={sentData} allEmoData={allEmoData} />

          <TimelineChart timeline={data.timeline} />

          <TopicBreakdown topics={data.topics} />

          {/* Language Breakdown */}
          {data.language_counts && Object.keys(data.language_counts).length > 1 && (
            <div className="card fade-in fade-in-delay-3">
              <div className="card-title">Languages detected</div>
              <div className="lang-grid">
                {Object.entries(data.language_counts).map(([lang, count]) => (
                  <div key={lang} className="lang-badge">
                    <div className="lang-count">{count}</div>
                    <div className="lang-name">{lang}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <CreatorBrief 
            suggestions={data.suggestions} 
            pinSuggestions={data.pin_suggestions} 
          />

          {/* Export Bar */}
          <div className="export-bar fade-in fade-in-delay-4">
            <button className="btn-export" onClick={() => exportCSV(data.comments)}>📥 Export CSV</button>
            <button className="btn-export" onClick={() => exportJSON(data)}>📄 Export JSON</button>
          </div>

          {/* Filters + Search */}
          <div className="filter-bar">
            {["all", "positive", "neutral", "negative", "mixed", "toxic"].map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`filter-btn ${filter === f ? "active" : ""}`}>
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
            <input className="search-input" placeholder="Search comments…" value={search}
              onChange={e => setSearch(e.target.value)} />
          </div>

          <CommentTable comments={visibleComments} />
        </>
      )}

      {/* ── Channel Results ─────────────────────────────────────── */}
      {activeTab === "channel" && channelData && !channelLoading && (
        <ChannelView data={channelData} setUrl={setUrl} setActiveTab={setActiveTab} />
      )}
      
    </div>
  );
}