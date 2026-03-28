type ChatConfig = {
  astraApiEndpoint: string;
  astraToken: string;
  astraNamespace: string;
  astraCollection: string;
  chatHistoryCollection: string;
  enableChatHistory: boolean;
  geminiApiKey: string;
  geminiModel: string;
  geminiApiVersion: "v1" | "v1beta";
  embedModel: string;
  embedDimension: number;
  geminiMaxOutputTokens: number;
  geminiTemperature: number;
  searchSimilarityThreshold: number;
  searchCoverageThreshold: number;
  searchContextDocs: number;
  searchPerSectionLimit: number;
  cacheMaxEntries: number;
  retrievalCacheTtlMs: number;
  embeddingCacheTtlMs: number;
  responseCacheTtlMs: number;
  rateLimitWindowMs: number;
  rateLimitMaxRequests: number;
  upstashUrl: string;
  upstashToken: string;
  experimentContext: string;
};

const DEFAULTS = {
  astraCollection: "experiment_docs",
  chatHistoryCollection: "chat_history",
  geminiModel: "gemini-2.5-flash-lite",
  embedModel: "text-embedding-004",
  embedDimension: 768,
  geminiMaxOutputTokens: 320,
  geminiTemperature: 0.2,
  searchSimilarityThreshold: 0.58,
  searchCoverageThreshold: 0.2,
  searchContextDocs: 4,
  searchPerSectionLimit: 6,
  cacheMaxEntries: 400,
  retrievalCacheTtlMs: 15_000,
  embeddingCacheTtlMs: 5 * 60_000,
  responseCacheTtlMs: 10_000,
  rateLimitWindowMs: 60_000,
  rateLimitMaxRequests: 30,
} as const;

let cachedConfig: ChatConfig | null = null;

function required(name: string): string {
  const value = String(process.env[name] ?? "").trim();
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

function parseIntEnv(name: string, fallback: number): number {
  const parsed = Number.parseInt(String(process.env[name] ?? ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseFloatEnv(name: string, fallback: number): number {
  const parsed = Number.parseFloat(String(process.env[name] ?? ""));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseBoolEnv(name: string, fallback: boolean): boolean {
  const value = String(process.env[name] ?? "").trim().toLowerCase();
  if (!value) return fallback;
  if (["1", "true", "yes", "on"].includes(value)) return true;
  if (["0", "false", "no", "off"].includes(value)) return false;
  return fallback;
}

export function getChatConfig(): ChatConfig {
  if (cachedConfig) return cachedConfig;

  const geminiApiKey = String(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "").trim();
  if (!geminiApiKey) {
    throw new Error("Missing GEMINI_API_KEY (or GOOGLE_API_KEY).");
  }

  const geminiApiVersion = String(process.env.GEMINI_API_VERSION ?? "v1").trim().toLowerCase() === "v1beta" ? "v1beta" : "v1";

  cachedConfig = {
    astraApiEndpoint: required("ASTRA_DB_API_ENDPOINT"),
    astraToken: required("ASTRA_DB_APPLICATION_TOKEN"),
    astraNamespace: required("ASTRA_DB_NAMESPACE"),
    astraCollection: String(process.env.ASTRA_DB_COLLECTION ?? DEFAULTS.astraCollection).trim() || DEFAULTS.astraCollection,
    chatHistoryCollection:
      String(process.env.ASTRA_DB_CHAT_HISTORY_COLLECTION ?? DEFAULTS.chatHistoryCollection).trim() ||
      DEFAULTS.chatHistoryCollection,
    enableChatHistory: parseBoolEnv("ENABLE_CHAT_HISTORY_STORAGE", false),
    geminiApiKey,
    geminiModel: String(process.env.GEMINI_MODEL ?? DEFAULTS.geminiModel).trim() || DEFAULTS.geminiModel,
    geminiApiVersion,
    embedModel: String(process.env.EMBED_MODEL ?? DEFAULTS.embedModel).trim() || DEFAULTS.embedModel,
    embedDimension: parseIntEnv("EMBED_DIM", DEFAULTS.embedDimension),
    geminiMaxOutputTokens: parseIntEnv("GEMINI_MAX_OUTPUT_TOKENS", DEFAULTS.geminiMaxOutputTokens),
    geminiTemperature: parseFloatEnv("GEMINI_TEMPERATURE", DEFAULTS.geminiTemperature),
    searchSimilarityThreshold: parseFloatEnv("SEARCH_SIM_THRESHOLD", DEFAULTS.searchSimilarityThreshold),
    searchCoverageThreshold: parseFloatEnv("SEARCH_KEYWORD_COVERAGE_THRESHOLD", DEFAULTS.searchCoverageThreshold),
    searchContextDocs: parseIntEnv("SEARCH_CONTEXT_DOCS", DEFAULTS.searchContextDocs),
    searchPerSectionLimit: parseIntEnv("SEARCH_PER_SECTION_LIMIT", DEFAULTS.searchPerSectionLimit),
    cacheMaxEntries: parseIntEnv("CACHE_MAX_ENTRIES", DEFAULTS.cacheMaxEntries),
    retrievalCacheTtlMs: parseIntEnv("CACHE_RETRIEVAL_TTL_MS", DEFAULTS.retrievalCacheTtlMs),
    embeddingCacheTtlMs: parseIntEnv("CACHE_EMBEDDING_TTL_MS", DEFAULTS.embeddingCacheTtlMs),
    responseCacheTtlMs: parseIntEnv("CACHE_RESPONSE_TTL_MS", DEFAULTS.responseCacheTtlMs),
    rateLimitWindowMs: parseIntEnv("RATE_LIMIT_WINDOW_MS", DEFAULTS.rateLimitWindowMs),
    rateLimitMaxRequests: parseIntEnv("RATE_LIMIT_MAX_REQUESTS", DEFAULTS.rateLimitMaxRequests),
    upstashUrl: String(process.env.UPSTASH_REDIS_REST_URL ?? "").trim(),
    upstashToken: String(process.env.UPSTASH_REDIS_REST_TOKEN ?? "").trim(),
    experimentContext: String(process.env.EXTERNAL_EXPERIMENT_CONTEXT ?? "").trim() || "this experiment",
  };

  return cachedConfig;
}
