# Virtual Lab Assistant (Production Refactor)

Next.js App Router chatbot with AstraDB vector search and Gemini generation, refactored for streaming, modularity, and production controls.

## Stack

- Next.js 16 + React 19
- AstraDB (`@datastax/astra-db-ts`)
- Gemini (`@google/generative-ai`)
- Markdown rendering (`react-markdown`)

## New server architecture

Core modules now live under `lib/server/chat/`:

- `config.ts`: centralized env validation and defaults
- `validate.ts`: request sanitization + payload limits
- `rate-limit.ts`: rate limiting (in-memory + optional Upstash Redis)
- `astra.ts`: Astra singleton client and query helpers
- `embeddings.ts`: Gemini embedding singleton + TTL cache
- `retrieval.ts`: direct-answer fast paths + section-aware vector retrieval
- `prompt.ts`: compact, cost-aware prompt builder
- `gemini.ts`: token streaming from Gemini
- `engine.ts`: orchestration (confidence checks + response caching)
- `stream.ts`: NDJSON streaming helper
- `history.ts`: optional async chat turn persistence

`app/api/chat/route.ts` is now a thin route (validation, rate limit, stream response).

## Streaming protocol

`POST /api/chat` returns `application/x-ndjson` by default:

- `{"type":"meta", ...}`
- `{"type":"delta","value":"..."}` (repeated)
- `{"type":"done", ...}`
- `{"type":"error","message":"..."}`

Client implementation in `app/page.tsx` incrementally renders assistant tokens.

To disable streaming per request:

```json
{ "question": "...", "stream": false }
```

## Environment variables

Required:

- `ASTRA_DB_API_ENDPOINT`
- `ASTRA_DB_APPLICATION_TOKEN`
- `ASTRA_DB_NAMESPACE`
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

Common optional:

- `ASTRA_DB_COLLECTION` (default: `experiment_docs`)
- `ASTRA_DB_CHAT_HISTORY_COLLECTION` (default: `chat_history`)
- `ENABLE_CHAT_HISTORY_STORAGE` (default: `false`)
- `GEMINI_MODEL` (default: `gemini-2.5-flash-lite`)
- `GEMINI_API_VERSION` (`v1` default, `v1beta` optional)
- `EMBED_MODEL` (default: `text-embedding-004`)
- `EMBED_DIM` (default: `768`)
- `GEMINI_MAX_OUTPUT_TOKENS` (default: `320`)
- `GEMINI_TEMPERATURE` (default: `0.2`)
- `SEARCH_SIM_THRESHOLD` (default: `0.58`)
- `SEARCH_KEYWORD_COVERAGE_THRESHOLD` (default: `0.2`)
- `SEARCH_CONTEXT_DOCS` (default: `4`)
- `SEARCH_PER_SECTION_LIMIT` (default: `6`)
- `CACHE_MAX_ENTRIES` (default: `400`)
- `CACHE_RETRIEVAL_TTL_MS` (default: `60000`)
- `CACHE_EMBEDDING_TTL_MS` (default: `300000`)
- `CACHE_RESPONSE_TTL_MS` (default: `45000`)
- `RATE_LIMIT_WINDOW_MS` (default: `60000`)
- `RATE_LIMIT_MAX_REQUESTS` (default: `30`)

Optional distributed rate limiting:

- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`

## Run

```bash
npm install
npm run seed
npm run dev
```

## Astra schema and indexing guidance

Use these document categories in your vector collection:

- `type: "text"` with `sectionId`, `sectionLabel`, `sourceFile`, `$vector`
- `type: "glossary"` with `key`, `display`, `$vector`
- `type: "rating"` with `key`, `display`, `$vector`
- `type: "assessment"` with `stage`, `number`, `answer`, `block`, `$vector`
- optional `type: "image"` with `path`, `sectionId`, `$vector`

For large scale, keep metadata low-cardinality and filterable (`type`, `sectionId`, `stage`, `number`, `key`) and ensure those fields are indexed in your Astra deployment strategy.

## Production recommendations

- Use Redis-backed rate limiting and caching for multi-instance deployments.
- Keep embeddings and generation models in singletons to reduce cold-request overhead.
- Keep prompts/context compact to reduce Gemini token cost.
- Prefer direct-answer metadata (`assessment`, `rating`, `glossary`) before LLM calls.
- Enforce strict request bounds and sanitize all input.
- Store chat history asynchronously and make it optional.

## Deployment (Vercel / Edge notes)

- Current route runs in `nodejs` runtime because Astra SDK and Gemini SDK require Node APIs.
- For Vercel:
  - set Node to `22.x`
  - configure all env vars in project settings
  - use region close to Astra + Gemini endpoints
- For high concurrency:
  - move in-memory cache/rate limit to Redis
  - add autoscaling and observability (p95 latency, token usage, retrieval hit rate)
