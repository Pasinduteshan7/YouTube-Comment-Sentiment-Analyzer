"""
AI-powered suggestion generation using Groq LLM.
Generates a strategic creator brief based on comment analysis data.
"""

import os
import groq


def generate_suggestions(
    comments: list,
    sentiment_counts: dict,
    emotion_counts: dict,
    fingerprint: dict = None,
    conflicted: list = None,
    like_weighted: dict = None,
):
    """Generates an AI-powered strategic brief for the content creator using Groq."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return [{"type": "warning", "title": "API Key Missing", "detail": "Add GROQ_API_KEY to your .env to enable the AI creator brief."}]

    try:
        client = groq.Groq(api_key=api_key)

        total = max(len(comments), 1)
        top_comments = sorted(comments, key=lambda x: int(x.get("likes", 0)), reverse=True)[:40]
        comment_texts = "\n".join([f"- (Likes: {c.get('likes', 0)}) {c.get('text', '')}" for c in top_comments])

        prompt = f"""You are an expert YouTube Strategist. Your client has just uploaded a video (or you are analysing their channel).
Based on the following data, write a 3-4 paragraph strategic brief for the creator.
Focus on specific insights, viewer demands, and actionable advice. DO NOT use generic advice. 
Reference specific viewer comments if relevant (e.g., "Several viewers asked for a Luke Cage crossover").
Do not include a greeting or sign-off, just output the brief in Markdown format.

DATA:
Total Comments Analysed: {total}
Sentiment: {sentiment_counts}
Emotions: {emotion_counts}
Emotional Fingerprint: {fingerprint.get('profile', 'Unknown') if fingerprint else 'Unknown'}

Top Comments:
{comment_texts}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert YouTube Strategist. Output markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1024,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq API Error: {e}")
        return [{"type": "warning", "title": "AI Brief Failed", "detail": f"Could not generate report: {str(e)}"}]
