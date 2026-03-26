import { getChatConfig } from "@/lib/server/chat/config";
import type { RateLimitState } from "@/lib/server/chat/types";

type MemoryBucket = {
  count: number;
  resetAt: number;
};

const memoryBuckets = new Map<string, MemoryBucket>();

function fromMemory(key: string): RateLimitState {
  const cfg = getChatConfig();
  const now = Date.now();
  const existing = memoryBuckets.get(key);

  if (!existing || now >= existing.resetAt) {
    memoryBuckets.set(key, { count: 1, resetAt: now + cfg.rateLimitWindowMs });
    return {
      allowed: true,
      limit: cfg.rateLimitMaxRequests,
      remaining: cfg.rateLimitMaxRequests - 1,
      retryAfterMs: 0,
      source: "memory",
    };
  }

  existing.count += 1;
  const allowed = existing.count <= cfg.rateLimitMaxRequests;
  return {
    allowed,
    limit: cfg.rateLimitMaxRequests,
    remaining: Math.max(0, cfg.rateLimitMaxRequests - existing.count),
    retryAfterMs: allowed ? 0 : Math.max(0, existing.resetAt - now),
    source: "memory",
  };
}

async function fromUpstash(key: string): Promise<RateLimitState | null> {
  const cfg = getChatConfig();
  if (!cfg.upstashUrl || !cfg.upstashToken) return null;

  const bucketKey = `rl:chat:${key}`;
  const body = JSON.stringify([
    ["INCR", bucketKey],
    ["PEXPIRE", bucketKey, cfg.rateLimitWindowMs, "NX"],
    ["PTTL", bucketKey],
  ]);

  try {
    const res = await fetch(`${cfg.upstashUrl}/pipeline`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${cfg.upstashToken}`,
        "Content-Type": "application/json",
      },
      body,
      cache: "no-store",
    });

    if (!res.ok) return null;

    const json = (await res.json()) as Array<{ result: unknown }> | unknown;
    const entries = Array.isArray(json) ? json : [];
    const count = Number(entries[0]?.result ?? 0);
    const pttl = Number(entries[2]?.result ?? cfg.rateLimitWindowMs);

    if (!Number.isFinite(count)) return null;
    return {
      allowed: count <= cfg.rateLimitMaxRequests,
      limit: cfg.rateLimitMaxRequests,
      remaining: Math.max(0, cfg.rateLimitMaxRequests - count),
      retryAfterMs: count <= cfg.rateLimitMaxRequests ? 0 : Math.max(0, Number.isFinite(pttl) ? pttl : cfg.rateLimitWindowMs),
      source: "upstash",
    };
  } catch {
    return null;
  }
}

export async function checkRateLimit(key: string): Promise<RateLimitState> {
  const remote = await fromUpstash(key);
  if (remote) return remote;
  return fromMemory(key);
}
