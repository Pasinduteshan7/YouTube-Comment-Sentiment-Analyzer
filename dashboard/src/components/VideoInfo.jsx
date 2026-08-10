import { formatNum } from "../constants";

export default function VideoInfo({ vi, total }) {
  if (!vi.title) return null;
  return (
    <div className="card video-info-card fade-in">
      {vi.thumbnail && <img src={vi.thumbnail} alt="thumbnail" className="video-info-thumb" />}
      <div className="video-info-details">
        <div className="video-info-title">{vi.title}</div>
        <div className="video-info-channel">{vi.channel} · {vi.published}</div>
        <div className="video-stats">
          {[
            { label: "Views", val: formatNum(vi.view_count) },
            { label: "Likes", val: formatNum(vi.like_count) },
            { label: "Comments", val: formatNum(vi.comment_count) },
            { label: "Analysed", val: total },
          ].map(s => (
            <div key={s.label} className="video-stat">
              <div className="video-stat-value">{s.val}</div>
              <div className="video-stat-label">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
