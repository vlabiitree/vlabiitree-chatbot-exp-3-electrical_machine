import { GoogleGenerativeAI, type EmbedContentRequest, TaskType } from "@google/generative-ai";
import { TTLCache } from "@/lib/server/chat/cache";
import { getChatConfig } from "@/lib/server/chat/config";
import { normalizeSpace } from "@/lib/server/chat/intent";

type Embedder = {
  embed: (text: string) => Promise<number[]>;
};

type ApiVersion = "v1" | "v1beta";
type EmbedModelClient = ReturnType<GoogleGenerativeAI["getGenerativeModel"]>;
type EmbedCandidate = {
  model: string;
  apiVersion: ApiVersion;
};

let embedderPromise: Promise<Embedder> | null = null;
let embedCache: TTLCache<number[]> | null = null;

function normalizeVector(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((acc, item) => acc + item * item, 0));
  if (!Number.isFinite(norm) || norm === 0) return vec;
  return vec.map((item) => item / norm);
}

function cacheKey(input: string): string {
  return normalizeSpace(input).toLowerCase();
}

function makeEmbedReq(text: string, dim: number): EmbedContentRequest {
  return {
    content: { role: "user", parts: [{ text }] },
    taskType: TaskType.RETRIEVAL_QUERY,
    outputDimensionality: dim,
  } as unknown as EmbedContentRequest;
}

function buildCandidates(preferredModel: string, preferredVersion: ApiVersion): EmbedCandidate[] {
  const candidates: EmbedCandidate[] = [
    { model: preferredModel, apiVersion: preferredVersion },
    { model: "text-embedding-004", apiVersion: "v1" },
    { model: "text-embedding-004", apiVersion: "v1beta" },
    { model: "gemini-embedding-001", apiVersion: "v1beta" },
  ];

  const unique: EmbedCandidate[] = [];
  const seen = new Set<string>();
  for (const cand of candidates) {
    const key = `${cand.model}@${cand.apiVersion}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(cand);
  }
  return unique;
}

async function probeCandidate(
  client: EmbedModelClient,
  dim: number
): Promise<{ ok: boolean; message?: string }> {
  try {
    const probe = await client.embedContent(makeEmbedReq("ping", dim));
    const values = probe?.embedding?.values as number[] | undefined;
    if (!values?.length) return { ok: false, message: "No embedding values returned." };
    if (values.length !== dim) {
      return { ok: false, message: `Dimension mismatch: got ${values.length}, expected ${dim}.` };
    }
    return { ok: true };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message };
  }
}

async function buildEmbedder(): Promise<Embedder> {
  const cfg = getChatConfig();
  const genAI = new GoogleGenerativeAI(cfg.geminiApiKey);

  const candidates = buildCandidates(cfg.embedModel, cfg.geminiApiVersion);
  let selected: EmbedModelClient | null = null;
  const failed: string[] = [];

  for (const cand of candidates) {
    const modelClient = genAI.getGenerativeModel(
      { model: cand.model },
      { apiVersion: cand.apiVersion }
    );
    const probe = await probeCandidate(modelClient, cfg.embedDimension);
    if (!probe.ok) {
      failed.push(`${cand.model}@${cand.apiVersion}: ${probe.message || "unknown error"}`);
      continue;
    }
    selected = modelClient;
    break;
  }

  if (!selected) {
    throw new Error(
      `No embedding model is available for this API key. Tried ${candidates
        .map((cand) => `${cand.model}@${cand.apiVersion}`)
        .join(", ")}. Failures: ${failed.join(" | ")}`
    );
  }

  return {
    embed: async (text: string) => {
      const response = await selected.embedContent(makeEmbedReq(text, cfg.embedDimension));
      const values = response?.embedding?.values as number[] | undefined;
      if (!values?.length) throw new Error("Embedding response was empty.");
      if (values.length !== cfg.embedDimension) {
        throw new Error(`Embedding dimension mismatch: got ${values.length}, expected ${cfg.embedDimension}.`);
      }
      // For dimensions lower than the native model output, normalize to keep cosine scores stable.
      return cfg.embedDimension === 3072 ? values : normalizeVector(values);
    },
  };
}

async function getEmbedder(): Promise<Embedder> {
  if (!embedderPromise) {
    embedderPromise = buildEmbedder().catch((error) => {
      embedderPromise = null;
      throw error;
    });
  }
  return embedderPromise;
}

export async function embedQuery(input: string): Promise<number[]> {
  const cfg = getChatConfig();
  if (!embedCache) {
    embedCache = new TTLCache<number[]>(cfg.cacheMaxEntries, cfg.embeddingCacheTtlMs);
  }

  const key = cacheKey(input);
  const cached = embedCache.get(key);
  if (cached) return cached;

  const embedder = await getEmbedder();
  const vector = await embedder.embed(input);
  embedCache.set(key, vector);
  return vector;
}
