import { TTLCache } from "@/lib/server/chat/cache";
import { getChatConfig } from "@/lib/server/chat/config";
import { NO_ANSWER_MESSAGE } from "@/lib/server/chat/constants";
import { streamGeminiAnswer } from "@/lib/server/chat/gemini";
import { lexicalOverlapScore, normalizeSpace } from "@/lib/server/chat/intent";
import { queryCoverage, retrieveContext } from "@/lib/server/chat/retrieval";
import type { ChatExecution, ChatRequest } from "@/lib/server/chat/types";

type ExecuteOptions = {
  onToken?: (value: string) => void;
};

let responseCache: TTLCache<ChatExecution> | null = null;

function cacheKey(req: ChatRequest): string {
  return normalizeSpace(req.question).toLowerCase();
}

function isRateLimitedError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const typed = error as { status?: number; message?: string };
  const message = String(typed.message ?? "").toLowerCase();
  return (
    typed.status === 429 ||
    message.includes("429") ||
    message.includes("quota exceeded") ||
    message.includes("too many requests")
  );
}

function buildExtractiveFallback(question: string, context: string, sources: string[]): string {
  const q = normalizeSpace(question).toLowerCase();
  const wantsAim = /\b(aim|objective|objectives)\b/.test(q);

  const candidates = [
    ...sources.map((s) => normalizeSpace(s)),
    ...normalizeSpace(context)
      .split(/\n\n---\n\n/)
      .map((chunk) => normalizeSpace(chunk)),
  ].filter(Boolean);

  if (!candidates.length) return NO_ANSWER_MESSAGE;

  const best = candidates
    .map((candidate) => ({
      candidate,
      score: (() => {
        let score = lexicalOverlapScore(question, candidate);
        if (wantsAim) {
          if (/\baim\b/i.test(candidate)) score += 1.4;
          if (/^\s*aim\b/i.test(candidate)) score += 0.8;
          if (/\banswer\s*:/i.test(candidate)) score -= 0.9;
        }
        return score;
      })(),
    }))
    .sort((a, b) => b.score - a.score)[0]?.candidate;

  if (!best) return NO_ANSWER_MESSAGE;

  let answer = best;
  if (/^\s*aim\b/i.test(answer)) {
    answer = answer
      .replace(/^\s*aim\b[\s:.-]*/i, "")
      .replace(/\btheory\b[\s\S]*$/i, "")
      .trim();
  }
  if (answer.length > 520) answer = `${answer.slice(0, 517)}...`;
  return answer || NO_ANSWER_MESSAGE;
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

  let answer: string;

  try {
    answer = await streamGeminiAnswer({
      question: req.question,
      context: retrieved.context,
      history: req.history,
      onToken: options.onToken,
    });
  } catch (error) {
    const fallback = buildExtractiveFallback(req.question, retrieved.context, retrieved.sources);
    const quotaLimited = isRateLimitedError(error);
    const fallbackOutput: ChatExecution = {
      answer: fallback,
      pathway: quotaLimited ? "fallback_rate_limited" : "fallback_llm_error",
      sources: retrieved.sources,
      similarity: retrieved.similarity,
      coverage,
    };
    options.onToken?.(fallbackOutput.answer);
    if (req.history.length === 0) responseCache.set(key, fallbackOutput);
    return fallbackOutput;
  }

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
