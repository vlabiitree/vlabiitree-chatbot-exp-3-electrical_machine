import { TTLCache } from "@/lib/server/chat/cache";
import { getChatConfig } from "@/lib/server/chat/config";
import { NO_ANSWER_MESSAGE } from "@/lib/server/chat/constants";
import { streamGeminiAnswer } from "@/lib/server/chat/gemini";
import { normalizeSpace } from "@/lib/server/chat/intent";
import { queryCoverage, retrieveContext } from "@/lib/server/chat/retrieval";
import type { ChatExecution, ChatRequest } from "@/lib/server/chat/types";

type ExecuteOptions = {
  onToken?: (value: string) => void;
};

let responseCache: TTLCache<ChatExecution> | null = null;

function cacheKey(req: ChatRequest): string {
  return normalizeSpace(req.question).toLowerCase();
}

export async function executeChat(req: ChatRequest, options: ExecuteOptions = {}): Promise<ChatExecution> {
  const cfg = getChatConfig();
  if (!responseCache) {
    responseCache = new TTLCache<ChatExecution>(cfg.cacheMaxEntries, cfg.responseCacheTtlMs);
  }

  const key = cacheKey(req);
  if (req.history.length === 0) {
    const hit = responseCache.get(key);
    if (hit) {
      options.onToken?.(hit.answer);
      return hit;
    }
  }

  const retrieved = await retrieveContext(req.question);
  const coverage = queryCoverage(req.question, retrieved.sources);

  if (retrieved.directAnswer) {
    const direct: ChatExecution = {
      answer: retrieved.directAnswer,
      pathway: retrieved.pathway,
      sources: retrieved.sources,
      similarity: retrieved.similarity,
      coverage,
    };
    if (req.history.length === 0) responseCache.set(key, direct);
    options.onToken?.(direct.answer);
    return direct;
  }

  const lowCoverage =
    coverage < cfg.searchCoverageThreshold &&
    retrieved.similarity < Math.max(cfg.searchSimilarityThreshold + 0.1, 0.72);
  const lowConfidence =
    !retrieved.context.trim() || retrieved.similarity < cfg.searchSimilarityThreshold || lowCoverage;

  if (lowConfidence) {
    const unknown: ChatExecution = {
      answer: NO_ANSWER_MESSAGE,
      pathway: "unknown_low_confidence",
      sources: retrieved.sources,
      similarity: retrieved.similarity,
      coverage,
    };
    options.onToken?.(unknown.answer);
    if (req.history.length === 0) responseCache.set(key, unknown);
    return unknown;
  }

  const answer = await streamGeminiAnswer({
    question: req.question,
    context: retrieved.context,
    history: req.history,
    onToken: options.onToken,
  });

  const output: ChatExecution = {
    answer,
    pathway: "llm_stream",
    sources: retrieved.sources,
    similarity: retrieved.similarity,
    coverage,
  };
  if (req.history.length === 0) responseCache.set(key, output);
  return output;
}
