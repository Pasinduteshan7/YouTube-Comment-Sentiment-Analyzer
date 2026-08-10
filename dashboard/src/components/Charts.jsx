import { useState } from "react";
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";
import { SENT_COLORS, EMO_COLORS } from "../constants";

export default function Charts({ sentData, allEmoData }) {
  const [showAllEmo, setShowAllEmo] = useState(false);
  const emoData = showAllEmo ? allEmoData : allEmoData.slice(0, 10);

  return (
    <div className="charts-grid fade-in fade-in-delay-2">
      {/* Sentiment Pie Chart */}
      <div className="card">
        <div className="card-title">Sentiment split</div>
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

      {/* Emotion Bar Chart */}
      <div className="card">
        <div className="chart-header">
          <span className="card-title" style={{ margin: 0 }}>Emotion breakdown</span>
          {allEmoData.length > 10 && (
            <button className="toggle-btn" onClick={() => setShowAllEmo(o => !o)}>
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
  );
}
