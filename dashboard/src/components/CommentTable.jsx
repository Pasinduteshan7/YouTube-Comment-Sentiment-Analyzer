import { useState } from "react";
import { EmotionBadges, SentimentBadge } from "./Badges";

const PER_PAGE = 50;

export default function CommentTable({ comments, title }) {
  const [page, setPage] = useState(1);

  const totalPages = Math.ceil(comments.length / PER_PAGE);
  const startIdx = (page - 1) * PER_PAGE;
  const pageComments = comments.slice(startIdx, startIdx + PER_PAGE);

  // Reset to page 1 if the comment list changes (e.g. filter applied)
  const prevLenRef = useState(comments.length);
  if (prevLenRef[0] !== comments.length) {
    prevLenRef[0] = comments.length;
    if (page > 1) setPage(1);
  }

  return (
    <div className="comment-table fade-in fade-in-delay-4">
      {title && (
        <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border-secondary)", fontWeight: 600, fontSize: "0.88rem" }}>
          {title}
        </div>
      )}

      <div className="comment-header">
        <span>Comment</span>
        <span style={{ textAlign: "center" }}>Sentiment</span>
        <span style={{ textAlign: "center" }}>Emotions</span>
        <span style={{ textAlign: "center" }}>Likes</span>
      </div>

      {pageComments.map((c, i) => (
        <div key={startIdx + i} className="comment-row">
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span className="comment-text">{c.text}</span>
            {c.video_title && (
              <span style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>from {c.video_title}</span>
            )}
          </div>
          <span style={{ textAlign: "center" }}><SentimentBadge c={c} /></span>
          <span style={{ textAlign: "center" }}><EmotionBadges emotions={c.emotions} /></span>
          <span className="comment-likes">{c.likes ?? "—"}</span>
        </div>
      ))}

      {comments.length === 0 && (
        <div className="comment-empty">No comments match your filter.</div>
      )}

      {/* Pagination Footer */}
      {comments.length > 0 && (
        <div className="pagination-footer">
          <div className="pagination-info">
            Showing {startIdx + 1}–{Math.min(startIdx + PER_PAGE, comments.length)} of {comments.length} comments
          </div>

          {totalPages > 1 && (
            <div className="pagination-controls">
              <button
                className="pagination-btn"
                disabled={page === 1}
                onClick={() => setPage(1)}
                title="First page"
              >
                ⟨⟨
              </button>
              <button
                className="pagination-btn"
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
              >
                ← Prev
              </button>

              {/* Page numbers */}
              {generatePageNumbers(page, totalPages).map((p, i) =>
                p === "..." ? (
                  <span key={`dots-${i}`} className="pagination-dots">…</span>
                ) : (
                  <button
                    key={p}
                    className={`pagination-btn ${p === page ? "active" : ""}`}
                    onClick={() => setPage(p)}
                  >
                    {p}
                  </button>
                )
              )}

              <button
                className="pagination-btn"
                disabled={page === totalPages}
                onClick={() => setPage(p => p + 1)}
              >
                Next →
              </button>
              <button
                className="pagination-btn"
                disabled={page === totalPages}
                onClick={() => setPage(totalPages)}
                title="Last page"
              >
                ⟩⟩
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Generates an array of page numbers with ellipsis for large page counts.
 * e.g. [1, 2, 3, "...", 8, 9, 10] or [1, "...", 4, 5, 6, "...", 10]
 */
function generatePageNumbers(current, total) {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }

  const pages = [];
  pages.push(1);

  if (current > 3) {
    pages.push("...");
  }

  const start = Math.max(2, current - 1);
  const end = Math.min(total - 1, current + 1);

  for (let i = start; i <= end; i++) {
    pages.push(i);
  }

  if (current < total - 2) {
    pages.push("...");
  }

  pages.push(total);
  return pages;
}
