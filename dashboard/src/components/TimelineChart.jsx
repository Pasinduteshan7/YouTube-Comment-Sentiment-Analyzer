import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend
} from "recharts";
import { SENT_COLORS } from "../constants";

export default function TimelineChart({ timeline }) {
  if (!timeline || timeline.length === 0) return null;

  return (
    <div className="card fade-in fade-in-delay-3" style={{ marginTop: '16px' }}>
      <div className="card-title">Sentiment Timeline</div>
      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '16px', marginTop: '-8px' }}>
        How the emotional reception of this video has evolved over time.
      </p>
      
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={timeline} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorPos" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={SENT_COLORS.positive} stopOpacity={0.8}/>
              <stop offset="95%" stopColor={SENT_COLORS.positive} stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorNeg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={SENT_COLORS.negative} stopOpacity={0.8}/>
              <stop offset="95%" stopColor={SENT_COLORS.negative} stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorNeu" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={SENT_COLORS.neutral} stopOpacity={0.8}/>
              <stop offset="95%" stopColor={SENT_COLORS.neutral} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-secondary)" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
          <Tooltip 
            contentStyle={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)', borderRadius: '8px' }}
            itemStyle={{ fontSize: '0.85rem' }}
            labelStyle={{ color: 'var(--text-primary)', marginBottom: '4px', fontWeight: 'bold' }}
          />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
          <Area type="monotone" dataKey="positive" stroke={SENT_COLORS.positive} fillOpacity={1} fill="url(#colorPos)" />
          <Area type="monotone" dataKey="negative" stroke={SENT_COLORS.negative} fillOpacity={1} fill="url(#colorNeg)" />
          <Area type="monotone" dataKey="neutral" stroke={SENT_COLORS.neutral} fillOpacity={1} fill="url(#colorNeu)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
