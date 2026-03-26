import { DataAPIClient } from "@datastax/astra-db-ts";
import { getChatConfig } from "@/lib/server/chat/config";

type AnyDoc = Record<string, unknown>;
type CursorLike = {
  limit?: (value: number) => CursorLike;
  includeSimilarity?: (value: boolean) => CursorLike;
  sort?: (value: Record<string, unknown>) => CursorLike;
  toArray: () => Promise<unknown>;
};

let client: DataAPIClient | null = null;
let db: ReturnType<DataAPIClient["db"]> | null = null;
const collections: Record<string, ReturnType<ReturnType<DataAPIClient["db"]>["collection"]>> = {};

function getCollection(name: string) {
  const cfg = getChatConfig();

  if (!client) client = new DataAPIClient(cfg.astraToken);
  if (!db) db = client.db(cfg.astraApiEndpoint, { keyspace: cfg.astraNamespace });
  if (!collections[name]) collections[name] = db.collection(name);

  return collections[name];
}

export function similarityFromDoc(doc: AnyDoc): number {
  const values = [doc.$similarity, doc.similarity, doc.score, doc.$score];
  for (const val of values) {
    if (typeof val === "number" && Number.isFinite(val)) return val;
  }
  return 0;
}

export async function findDocs(
  filter: Record<string, unknown>,
  options?: {
    limit?: number;
    projection?: Record<string, 0 | 1>;
    collectionName?: string;
    sort?: Record<string, 1 | -1>;
  }
): Promise<AnyDoc[]> {
  const cfg = getChatConfig();
  const coll = getCollection(options?.collectionName || cfg.astraCollection);

  try {
    let cursor = coll.find(filter as never, {
      limit: options?.limit,
      projection: options?.projection,
      sort: options?.sort,
    } as never) as unknown as CursorLike;
    if (typeof cursor.limit === "function" && options?.limit) cursor = cursor.limit(options.limit);
    const out = await cursor.toArray();
    return Array.isArray(out) ? out : [];
  } catch {
    return [];
  }
}

export async function vectorFind(
  filter: Record<string, unknown>,
  vector: number[],
  limit: number,
  options?: {
    projection?: Record<string, 0 | 1>;
    collectionName?: string;
  }
): Promise<AnyDoc[]> {
  const cfg = getChatConfig();
  const coll = getCollection(options?.collectionName || cfg.astraCollection);

  try {
    let cursor = coll.find(filter as never, {
      limit,
      projection: options?.projection,
    } as never) as unknown as CursorLike;
    if (typeof cursor.sort === "function") {
      cursor = cursor.sort({ $vector: vector });
    }
    if (typeof cursor.includeSimilarity === "function") cursor = cursor.includeSimilarity(true);
    if (typeof cursor.limit === "function") cursor = cursor.limit(limit);
    const out = await cursor.toArray();
    return Array.isArray(out) ? out : [];
  } catch {
    try {
      let cursor = coll.find(filter as never) as unknown as CursorLike;
      if (typeof cursor.sort === "function") {
        cursor = cursor.sort({ $vector: vector });
      }
      if (typeof cursor.includeSimilarity === "function") cursor = cursor.includeSimilarity(true);
      if (typeof cursor.limit === "function") cursor = cursor.limit(limit);
      const out = await cursor.toArray();
      return Array.isArray(out) ? out : [];
    } catch {
      return [];
    }
  }
}

export async function insertDoc(collectionName: string, doc: Record<string, unknown>): Promise<void> {
  try {
    await getCollection(collectionName).insertOne(doc as never);
  } catch {
    // best effort write path
  }
}
