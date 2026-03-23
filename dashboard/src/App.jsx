import { useState, useEffect } from "react";
import Papa from "papaparse";
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from "recharts";

const SENT_COLORS = { positive: "#1D9E75", neutral: "#888780", negative: "#E24B4A" };
const EMO_COLORS  = { joy:"#BA7517", neutral:"#888780", surprise:"#378ADD", anger:"#E24B4A", sadness:"#534AB7", fear:"#7F77DD", disgust:"#D85A30" };

export default function App() {
  const [comments, setComments] = useState([]);
  const [filter, setFilter]     = useState("all");
  const [search, setSearch]     = useState("");

  useEffect(() => {
    Papa.parse("/comments_analysed.csv", {
      download: true, header: true, skipEmptyLines: true,
      complete: (res) => setComments(res.data)
    });
  }, []);

  // ── counts ──
  const sentCount = (l) => comments.filter(c => c.sentiment === l).length;
  const emoCount  = (l) => comments.filter(c => c.emotion   === l).length;

  const sentData = [
    { name: "Positive", value: sentCount("positive") },
    { name: "Neutral",  value: sentCount("neutral")  },
    { name: "Negative", value: sentCount("negative") },
  ];
  const emoData = ["joy","neutral","surprise","anger","sadness","fear","disgust"]
    .map(e => ({ name: e, value: emoCount(e) }))
    .filter(e => e.value > 0);

  // ── filtered comments ──
  const visible = comments.filter(c => {
    const matchFilter = filter === "all" || c.sentiment === filter;
    const matchSearch = c.text?.toLowerCase().includes(search.toLowerCase());
    return matchFilter && matchSearch;
  });

  const total = comments.length;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto", padding: "2rem 1rem", color: "#1a1a1a" }}>

      {/* header */}
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 4 }}>YouTube Comment Sentiment Dashboard</h1>
      <p style={{ color: "#888", fontSize: 14, marginBottom: 28 }}>{total} comments analysed</p>

      {/* metric cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12, marginBottom: 28 }}>
        {[
          { label: "Positive", count: sentCount("positive"), color: "#1D9E75" },
          { label: "Neutral",  count: sentCount("neutral"),  color: "#888780" },
          { label: "Negative", count: sentCount("negative"), color: "#E24B4A" },
        ].map(m => (
          <div key={m.label} style={{ background: "#f5f5f3", borderRadius: 10, padding: "1rem", textAlign: "center" }}>
            <div style={{ fontSize: 28, fontWeight: 600, color: m.color }}>{m.count}</div>
            <div style={{ fontSize: 12, color: "#888", marginTop: 4 }}>{m.label} · {total ? Math.round(m.count/total*100) : 0}%</div>
          </div>
        ))}
      </div>

      {/* charts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 28 }}>

        {/* sentiment pie */}
        <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1rem" }}>
          <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 12 }}>Sentiment split</p>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={sentData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({name,percent})=>`${name} ${Math.round(percent*100)}%`} labelLine={false}>
                {sentData.map(e => <Cell key={e.name} fill={SENT_COLORS[e.name.toLowerCase()]}/>)}
              </Pie>
              <Tooltip/>
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* emotion bar */}
        <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, padding: "1rem" }}>
          <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 12 }}>Emotion breakdown</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={emoData} layout="vertical" margin={{ left: 10 }}>
              <XAxis type="number" tick={{ fontSize: 11 }}/>
              <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={60}/>
              <Tooltip/>
              <Bar dataKey="value" radius={[0,4,4,0]}>
                {emoData.map(e => <Cell key={e.name} fill={EMO_COLORS[e.name] || "#888"}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* filters + search */}
      <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap", alignItems: "center" }}>
        {["all","positive","neutral","negative"].map(f => (
          <button key={f} onClick={() => setFilter(f)} style={{
            padding: "5px 16px", borderRadius: 20, fontSize: 13, cursor: "pointer", fontWeight: 500,
            background: filter === f ? "#185FA5" : "transparent",
            color: filter === f ? "#fff" : "#555",
            border: filter === f ? "none" : "0.5px solid #ccc"
          }}>{f.charAt(0).toUpperCase()+f.slice(1)}</button>
        ))}
        <input
          placeholder="Search comments..."
          value={search} onChange={e => setSearch(e.target.value)}
          style={{ marginLeft: "auto", padding: "6px 12px", borderRadius: 8, border: "0.5px solid #ccc", fontSize: 13, width: 200 }}
        />
      </div>

      {/* comment table */}
      <div style={{ background: "#fff", border: "0.5px solid #e0e0e0", borderRadius: 12, overflow: "hidden" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 100px 100px", background: "#f5f5f3", padding: "8px 14px", fontSize: 12, fontWeight: 500, color: "#666" }}>
          <span>Comment</span><span style={{textAlign:"center"}}>Sentiment</span><span style={{textAlign:"center"}}>Emotion</span>
        </div>
        {visible.slice(0, 50).map((c, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 100px 100px", padding: "10px 14px", borderTop: "0.5px solid #f0f0f0", fontSize: 13, alignItems: "center" }}>
            <span style={{ color: "#333", lineHeight: 1.4 }}>{c.text}</span>
            <span style={{ textAlign: "center" }}>
              <span style={{ padding: "2px 8px", borderRadius: 8, fontSize: 11, fontWeight: 500,
                background: c.sentiment==="positive"?"#E1F5EE":c.sentiment==="negative"?"#FCEBEB":"#F1EFE8",
                color: c.sentiment==="positive"?"#085041":c.sentiment==="negative"?"#A32D2D":"#444" }}>
                {c.sentiment}
              </span>
            </span>
            <span style={{ textAlign: "center" }}>
              <span style={{ padding: "2px 8px", borderRadius: 8, fontSize: 11, fontWeight: 500, background: "#F1EFE8", color: "#444" }}>
                {c.emotion}
              </span>
            </span>
          </div>
        ))}
        {visible.length === 0 && (
          <div style={{ padding: "2rem", textAlign: "center", color: "#888", fontSize: 14 }}>No comments match your filter.</div>
        )}
        {visible.length > 50 && (
          <div style={{ padding: "10px", textAlign: "center", color: "#888", fontSize: 12 }}>Showing 50 of {visible.length} comments</div>
        )}
      </div>
    </div>
  );
}