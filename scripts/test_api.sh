#!/usr/bin/env bash
# ============================================================
# DocuMind API — Sample curl requests
# Run the server first: python main.py
# ============================================================

BASE="http://localhost:8000/api"

echo "════════════════════════════════════════"
echo "  1. Health Check"
echo "════════════════════════════════════════"
curl -s "$BASE/health" | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "  2. List Documents"
echo "════════════════════════════════════════"
curl -s "$BASE/documents" | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "  3. Submit a Query"
echo "════════════════════════════════════════"
curl -s -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I use dependency injection in FastAPI?",
    "session_id": "test_session_001"
  }' | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "  4. Query with Chat History"
echo "════════════════════════════════════════"
curl -s -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you show me an example?",
    "session_id": "test_session_001",
    "chat_history": [
      {"role": "user", "content": "How do I use dependency injection in FastAPI?"},
      {"role": "assistant", "content": "FastAPI uses the Depends() function..."}
    ]
  }' | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "  5. Ingest a URL"
echo "════════════════════════════════════════"
curl -s -X POST "$BASE/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://fastapi.tiangolo.com/tutorial/query-params/"}' \
  | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "  6. Ingest a File"
echo "════════════════════════════════════════"
curl -s -X POST "$BASE/ingest" \
  -F "file=@docs/fastapi_guide.md" \
  -F "title=FastAPI Guide" \
  | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "  7. Submit Feedback"
echo "════════════════════════════════════════"
curl -s -X POST "$BASE/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I use dependency injection in FastAPI?",
    "answer": "Use Depends()...",
    "rating": "up",
    "comment": "Very clear explanation!"
  }' | python3 -m json.tool

echo ""
echo "════════════════════════════════════════"
echo "  8. Feedback Stats"
echo "════════════════════════════════════════"
curl -s "$BASE/feedback/stats" | python3 -m json.tool
