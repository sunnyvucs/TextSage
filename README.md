# TextSage
A local-first RAG pipeline that turns School textbooks (PDF) into a queryable AI tutor. It extracts text and figures, aligns chapters to exact PDF pages, chunks content by topic, stores everything in PostgreSQL + Qdrant, and serves a streaming chat UI — all running on your own machine with no cloud dependency at inference time.
