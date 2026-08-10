export default function LoadingSkeleton({ type = "single" }) {
  return (
    <div className="loading-container fade-in">
      <div className="spinner" />
      <div className="loading-text">
        {type === "channel" ? "Scanning channel videos one by one…" : "Fetching comments and running AI models…"}
      </div>
      <div className="loading-hint">
        {type === "channel" ? "Each video takes about 20-30 seconds. Please wait." : "This takes about 20–40 seconds"}
      </div>
      <div style={{ marginTop: "2rem" }}>
        <div className="skeleton-row">
          {[1,2,3,4].map(i => <div key={i} className="skeleton skeleton-card" />)}
        </div>
        <div className="skeleton skeleton-chart" />
        <div className="skeleton skeleton-table" />
      </div>
    </div>
  );
}
