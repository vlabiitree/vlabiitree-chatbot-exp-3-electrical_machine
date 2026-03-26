// scripts/loadDB.ts
import fs from "fs";
import path from "path";
import { config as loadEnv } from "dotenv";
import { DataAPIClient } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAI, type EmbedContentRequest, TaskType } from "@google/generative-ai";
import crypto from "crypto";
import mammoth from "mammoth";
import { DOC_SECTIONS, DocSectionConfig } from "../config/docSections";

// --------------------
// Load env
// --------------------
(() => {
  const cwd = process.cwd();
  const candidates = [
    path.resolve(cwd, ".env"),
    path.resolve(__dirname, "../.env"),
    path.resolve(cwd, "vlab-chatbot/.env"),
    path.resolve(cwd, "vlab-chatbot-exp2/.env"),
  ];

  let loadedFrom: string | null = null;
  for (const p of candidates) {
    try {
      const res = loadEnv({ path: p });
      if (!res.error) {
        loadedFrom = p;
        break;
      }
    } catch {}
  }

  if (!loadedFrom) {
    const res = loadEnv();
    if (!res.error) loadedFrom = ".env (default lookup)";
  }

  if (!loadedFrom) console.warn("Warning: no .env file loaded; expecting env vars to be set.");
  else console.log(`Loaded env from: ${loadedFrom}`);
})();

const {
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  GEMINI_API_KEY,
  GEMINI_API_VERSION,
  EMBED_PROVIDER,
  EMBED_MODEL,
} = process.env as Record<string, string>;

if (!ASTRA_DB_API_ENDPOINT || !ASTRA_DB_APPLICATION_TOKEN || !ASTRA_DB_NAMESPACE) {
  throw new Error(
    "Missing Astra DB env vars. Required: ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_NAMESPACE"
  );
}
if (!ASTRA_DB_COLLECTION) throw new Error("Missing ASTRA_DB_COLLECTION in environment.");
if (!GEMINI_API_KEY) throw new Error("Missing GEMINI_API_KEY (needed for Google embeddings).");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const WORKDIR_CANDIDATES = Array.from(
  new Set([process.cwd(), PROJECT_ROOT, path.resolve(process.cwd(), "vlab-chatbot")])
);

type ResolvedSectionDoc = {
  section: DocSectionConfig;
  fullPath: string;
  relativePath: string;
};

const envKeyForSection = (id: string) =>
  `DOC_SECTION_${id.replace(/[^a-z0-9]+/gi, "_").toUpperCase()}_PATH`;

const resolveDocForSection = (section: DocSectionConfig): ResolvedSectionDoc | null => {
  const envKey = envKeyForSection(section.id);
  const override = process.env[envKey];
  const hints = (override ? [override] : []).concat(section.fileHints ?? []);

  for (const hintRaw of hints) {
    if (!hintRaw) continue;

    const candidates = path.isAbsolute(hintRaw)
      ? [hintRaw]
      : WORKDIR_CANDIDATES.map((dir) => path.resolve(dir, hintRaw));

    for (const candidate of candidates) {
      try {
        if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
          const relativePath = path.relative(PROJECT_ROOT, candidate).replace(/\\/g, "/");
          return { section, fullPath: candidate, relativePath };
        }
      } catch {}
    }
  }
  return null;
};

const resolveAllSectionDocs = () => {
  const found: ResolvedSectionDoc[] = [];
  const missing: DocSectionConfig[] = [];

  for (const section of DOC_SECTIONS) {
    const resolved = resolveDocForSection(section);
    if (resolved) found.push(resolved);
    else missing.push(section);
  }
  return { found, missing };
};

// --------------------
// Embeddings: GOOGLE only
// --------------------
const provider = (EMBED_PROVIDER || "google").toLowerCase();
if (provider !== "google") {
  throw new Error(
    `This loader is configured for EMBED_PROVIDER=google. You set "${provider}". Either change env or update loader accordingly.`
  );
}

const preferredApiVersion: "v1" | "v1beta" =
  (GEMINI_API_VERSION || "v1").trim().toLowerCase() === "v1" ? "v1" : "v1beta";

const embedModel = (EMBED_MODEL || "text-embedding-004").trim();
const embedDimEnv = Number.parseInt(
  process.env.EMBED_DIM || process.env.EMBED_DIMENSION || process.env.EMBED_OUTPUT_DIM || "",
  10
);
// default to 768 to keep collection size small; override via EMBED_DIM if desired
const embedDimension = Number.isFinite(embedDimEnv) && embedDimEnv > 0 ? embedDimEnv : 768;

const collectionName = ASTRA_DB_COLLECTION;

const normalizeVector = (vec: number[]): number[] => {
  const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  if (!norm || !Number.isFinite(norm)) return vec;
  return vec.map((v) => v / norm);
};

const makeEmbedReq = (text: string): EmbedContentRequest =>
  ({
    content: { role: "user", parts: [{ text }] },
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    outputDimensionality: embedDimension,
  } as unknown as EmbedContentRequest);

type EmbedModelResolution = {
  client: ReturnType<GoogleGenerativeAI["getGenerativeModel"]>;
  apiVersion: "v1" | "v1beta";
  model: string;
};

const resolveEmbedModel = async (): Promise<EmbedModelResolution> => {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const candidates: { model: string; apiVersion: "v1" | "v1beta" }[] = [
    { model: embedModel, apiVersion: preferredApiVersion },
    { model: "text-embedding-004", apiVersion: "v1" },
    { model: "text-embedding-004", apiVersion: "v1beta" },
    { model: "gemini-embedding-001", apiVersion: "v1beta" },
  ];

  let lastError: unknown = null;
  for (const cand of candidates) {
    try {
      const client = genAI.getGenerativeModel({ model: cand.model }, { apiVersion: cand.apiVersion });
      const probe = await client.embedContent(makeEmbedReq("ping"));
      const vec = probe?.embedding?.values as number[] | undefined;
      if (!vec?.length) throw new Error("No embedding values returned in probe.");
      return { client, apiVersion: cand.apiVersion, model: cand.model };
    } catch (err) {
      lastError = err;
      continue;
    }
  }
  throw new Error(
    `No supported embedding model available for this API key. Tried ${candidates
      .map((c) => `${c.model}@${c.apiVersion}`)
      .join(", ")}. Last error: ${String((lastError as Error)?.message || lastError)}`
  );
};

let embedModelClientPromise: Promise<EmbedModelResolution> | null = null;
const getEmbedModelClient = async () => {
  if (!embedModelClientPromise) {
    embedModelClientPromise = resolveEmbedModel();
  }
  return (await embedModelClientPromise).client;
};

const embed = async (text: string) => {
  const embedModelClient = await getEmbedModelClient();
  const res = await embedModelClient.embedContent(makeEmbedReq(text));

  const vec = res?.embedding?.values as number[] | undefined;
  if (!vec || !Array.isArray(vec) || !vec.length) {
    throw new Error("Failed to get embedding vector from Google API response");
  }
  if (vec.length !== embedDimension) {
    throw new Error(`Embedding dim mismatch (Google): got ${vec.length}, expected ${embedDimension}`);
  }
  // Gemini embeddings are only pre-normalized at 3072 dims; normalize for smaller sizes to improve cosine scores
  return embedDimension === 3072 ? vec : normalizeVector(vec);
};

// --------------------
// Astra client
// --------------------
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 700, chunkOverlap: 120 });

const normalize = (s: string) => s.replace(/[\r\t ]+/g, " ").replace(/\s*\n\s*/g, "\n").trim();
const hashId = (prefix: string, s: string) => `${prefix}_${crypto.createHash("sha1").update(s).digest("hex")}`;

const normKeyText = (s: string) =>
  (s || "")
    .toLowerCase()
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+/g, " ")
    .trim();

// DOCX reader
const readDocxFile = async (filePath: string): Promise<string> => {
  const header = Buffer.alloc(4);
  const fd = fs.openSync(filePath, "r");
  try {
    fs.readSync(fd, header, 0, 4, 0);
  } finally {
    fs.closeSync(fd);
  }
  const isZip = header[0] === 0x50 && header[1] === 0x4b; // 'PK'
  if (isZip) {
    const { value } = await mammoth.extractRawText({ path: filePath });
    return value;
  }
  return fs.readFileSync(filePath, "utf8");
};

function extractExperimentName(text: string): string | null {
  const t = normalize(text);
  const m1 = t.match(/Name of the experiment\s*(?:is)?\s*:\s*\n([^\n]+)\n?/i);
  if (m1?.[1]) return m1[1].trim();
  const m2 = t.match(/Experiment name\s*:\s*\n?([^\n]+)\n?/i);
  if (m2?.[1]) return m2[1].trim();
  const m3 = t.match(/Welcome to the experiment\s*[“"](.*?)[”"]/i);
  if (m3?.[1]) return m3[1].trim();
  return null;
}

// -------- META extractors (glossary + ratings) --------
function getBlock(text: string, startRe: RegExp, stopRe: RegExp, maxLines = 400): string | null {
  const lines = text.split(/\r?\n/).map((l) => l.trimEnd());
  const startIdx = lines.findIndex((l) => startRe.test(l.trim()));
  if (startIdx < 0) return null;

  const out: string[] = [];
  for (let i = startIdx + 1; i < lines.length; i++) {
    const t = lines[i].trim();
    if (stopRe.test(t)) break;
    if (out.length >= maxLines) break;
    out.push(lines[i].trimEnd());
  }
  const joined = normalize(out.join("\n"));
  return joined || null;
}

function extractGlossaryItems(docText: string): Array<{ term: string; text: string }> {
  const t = normalize(docText);

  // apparatus block ends at Ratings or About buttons
  const block =
    getBlock(
      t,
      /^(simulation\s*)?apparatus\b|^apparatus\s*-\s*the simulation includes/i,
      /^(ratings\b|about buttons\b|procedure\b|pretest\b|posttest\b|references\b)/i
    ) || "";

  if (!block) return [];

  const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
  const items: Array<{ term: string; text: string }> = [];

  let curTerm: string | null = null;
  let curLines: string[] = [];

  const flush = () => {
    if (!curTerm) return;
    const txt = normalize(curLines.join(" "));
    if (txt) items.push({ term: curTerm, text: txt });
    curTerm = null;
    curLines = [];
  };

  for (const line of lines) {
    const m = line.match(/^([A-Za-z0-9][A-Za-z0-9\s\-/()]+)\s*:\s*(.+)$/);
    if (m) {
      flush();
      curTerm = m[1].trim();
      curLines.push(m[2].trim());
      continue;
    }
    if (curTerm) curLines.push(line);
  }
  flush();

  return items;
}

function extractRatingsItems(docText: string): Array<{ term: string; value: string }> {
  const t = normalize(docText);

  const block =
    getBlock(
      t,
      /^ratings\b/i,
      /^(about buttons\b|procedure\b|simulation\b|pretest\b|posttest\b|references\b)/i
    ) || "";

  if (!block) return [];

  const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
  const out: Array<{ term: string; value: string }> = [];

  let curTerm: string | null = null;
  let curLines: string[] = [];

  const flush = () => {
    if (!curTerm) return;
    const value = normalize(curLines.join(" "));
    if (value) out.push({ term: curTerm, value });
    curTerm = null;
    curLines = [];
  };

  for (const line of lines) {
    const m = line.match(/^([^:]+)\s*:\s*(.+)$/);
    if (m) {
      flush();
      curTerm = m[1].trim();
      curLines.push(m[2].trim());
      continue;
    }
    if (curTerm) curLines.push(line.trim());
  }
  flush();

  return out;
}

type AssessmentStage = "pretest" | "posttest";

const normalizeInline = (s: string) =>
  (s || "").replace(/[\r\t ]+/g, " ").replace(/\s*\n\s*/g, " ").trim();

function extractAssessmentStageText(docText: string, stage: AssessmentStage): string | null {
  const t = normalize(docText);
  if (!t) return null;

  const lines = t.split("\n").map((l) => l.trimEnd());

  const startRe = stage === "pretest" ? /^pre\s*-?\s*test\b/i : /^post\s*-?\s*test\b/i;
  const stopRe = stage === "pretest" ? /^post\s*-?\s*test\b/i : null;

  const startIdx = lines.findIndex((l) => startRe.test(l.trim()));
  if (startIdx < 0) return null;

  const out: string[] = [];
  for (let i = startIdx + 1; i < lines.length; i++) {
    const cur = lines[i].trim();
    if (stopRe && stopRe.test(cur)) break;
    out.push(lines[i]);
  }

  const joined = normalize(out.join("\n"));
  return joined || null;
}

function extractAssessmentBlocks(stageText: string): Array<{ number: number; block: string }> {
  const t = normalize(stageText || "");
  if (!t) return [];

  const out: Array<{ number: number; block: string }> = [];

  const re = /(?:^|\n)\s*(\d{1,2})[.)]\s+([\s\S]*?)(?=(?:\n\s*\d{1,2}[.)]\s+)|$)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(t))) {
    const number = parseInt(String(m[1]), 10);
    if (!Number.isFinite(number)) continue;
    const block = normalize(m[0].trim());
    if (block) out.push({ number, block });
  }

  if (out.length) return out;

  // Fallback: use each "Answer:" line as a boundary and assign ordinal numbers.
  const lines = t.split(/\r?\n/);
  let cur: string[] = [];
  let ordinal = 1;

  const flush = () => {
    const block = normalize(cur.join("\n"));
    if (block) out.push({ number: ordinal++, block });
    cur = [];
  };

  for (const line of lines) {
    cur.push(line);
    if (/answer\s*:/i.test(line)) flush();
  }
  if (cur.length && !out.length) flush();

  return out;
}

function extractAssessmentQuestionLine(block: string): string {
  const body = normalize(block || "").replace(/^\s*\d{1,2}[.)]\s*/, "");
  const lines = body.split("\n").map((l) => l.trim()).filter(Boolean);
  return lines.length ? normalizeInline(lines[0]) : "";
}

function extractAssessmentAnswerOnly(block: string): string | null {
  const src = normalizeInline(block || "");
  if (!src) return null;

  const split = src.split(/\b(correct\s*answer|answer)\s*:/i);
  const questionPart = split[0] || "";

  const options: Record<string, string> = {};
  {
    const flat = normalizeInline(questionPart);
    const re = /([a-d])\s*[:)]\s*/gi;
    const rawHits: Array<{ letter: string; start: number; end: number }> = [];
    let m: RegExpExecArray | null;
    while ((m = re.exec(flat))) {
      rawHits.push({ letter: String(m[1]).toLowerCase(), start: m.index, end: m.index + m[0].length });
    }
    const hits: Array<{ letter: string; start: number; end: number }> = [];
    const seen = new Set<string>();
    for (const h of rawHits) {
      if (seen.has(h.letter)) continue;
      seen.add(h.letter);
      hits.push(h);
      if (seen.size >= 4) break;
    }
    for (let i = 0; i < hits.length; i++) {
      const cur = hits[i];
      const next = hits[i + 1];
      const raw = flat.slice(cur.end, next ? next.start : flat.length);
      const text = normalizeInline(raw);
      if (text) options[cur.letter] = text;
    }
  }

  const ansMatch =
    src.match(/\bcorrect\s*answer\s*:\s*(.+)$/i) || src.match(/\banswer\s*:\s*(.+)$/i);
  if (!ansMatch?.[1]) return null;

  let ans = ansMatch[1].trim();
  const explIdx = ans.search(/(?:because|explanation|reason)\b/i);
  if (explIdx > 0) ans = ans.slice(0, explIdx).trim();

  const mLetter = ans.match(/^\(?\s*([a-d])\s*\)?\s*[).:-]?\s*(.*)$/i);
  const letter = mLetter?.[1] ? String(mLetter[1]).toLowerCase() : null;

  if (letter && options[letter]) return options[letter];

  const rest = (mLetter?.[2] || ans).trim();
  if (!rest) return letter ? letter.toUpperCase() : null;

  const cleaned = rest.replace(/^\(?\s*[a-d]\s*\)?\s*[).:-]?\s*/i, "").trim();
  return cleaned || rest;
}

const createCollection = async (): Promise<void> => {
  const reset = String(process.env.SEED_RESET_COLLECTION ?? "").trim().toLowerCase();
  const shouldReset = reset === "1" || reset === "true" || reset === "yes";

  if (shouldReset) {
    try {
      console.log(`Dropping collection: ${collectionName}`);
      await db.collection(collectionName).drop();
    } catch (err: any) {
      console.log("Collection drop skipped:", err?.message || String(err));
    }
  }

  try {
    console.log(`Creating/using collection: ${collectionName} (dim=${embedDimension})`);
    const res = await db.createCollection(collectionName, {
      vector: { dimension: embedDimension, metric: "cosine" },
    });
    console.log("Collection created:", res);
  } catch (err: any) {
    console.log("Collection may already exist:", err?.message || String(err));
  }
};

const shouldCleanBySource = () => {
  const v = String(process.env.SEED_CLEAN ?? "true").trim().toLowerCase();
  return !(v === "0" || v === "false" || v === "no");
};

const cleanSourceDocs = async (collection: any, sourceFile: string) => {
  if (!shouldCleanBySource()) return;
  if (!sourceFile) return;

  try {
    const res: any = await collection.deleteMany({ sourceFile } as any);
    const deleted =
      typeof res?.deletedCount === "number"
        ? res.deletedCount
        : typeof res?.deleted === "number"
        ? res.deleted
        : null;
    console.log(`Cleared existing docs for ${sourceFile} (${deleted ?? "unknown"} deleted)`);
  } catch (e: any) {
    console.warn(`Failed to clear existing docs for ${sourceFile}:`, e?.message || e);
  }
};

const loadAndStoreEmbeddings = async (): Promise<void> => {
  const collection = db.collection(collectionName);

  const { found: resolvedSections, missing } = resolveAllSectionDocs();
  if (!resolvedSections.length) {
    throw new Error(
      "No DOCX files found. Put section DOCX under doc/ or docs/ or set DOC_SECTION_*_PATH overrides."
    );
  }

  const missingRequired = missing.filter((m) => !m.optional);
  if (missingRequired.length) {
    console.warn(
      "Missing DOCX files for sections:",
      missingRequired.map((m) => m.label).join(", ")
    );
  }

  let added = 0, updated = 0, skipped = 0, metaUpserts = 0;

  for (const entry of resolvedSections) {
    console.log(`Processing ${entry.section.label} from ${entry.relativePath}`);
    await cleanSourceDocs(collection, entry.relativePath);
    const docTextRaw = await readDocxFile(entry.fullPath);
    const docText = normalize(docTextRaw);

    // ---- meta: experiment name
    const expName = extractExperimentName(docText);
    if (expName) {
      const metaId = hashId("meta_experiment_name", expName);
      try {
        const vec = await embed(`experiment name ${expName}`);
        const metaDoc: any = {
          _id: metaId,
          uid: metaId,
          type: "meta",
          key: "experiment_name",
          text: expName,
          display: expName,
          sourceFile: entry.relativePath,
          sectionId: entry.section.id,
          sectionLabel: entry.section.label,
          $vector: vec,
          model: embedModel,
        };
        await collection.replaceOne({ _id: metaId } as any, metaDoc, { upsert: true } as any);
        metaUpserts++;
      } catch (e: any) {
        console.warn("Failed to upsert experiment name meta:", e?.message || e);
      }
    }

    // ---- meta: glossary items (apparatus definitions)
    const glossary = extractGlossaryItems(docText);
    for (const g of glossary) {
      const key = normKeyText(g.term);
      const id = hashId("glossary", `${key}::${g.text}`);
      try {
        const vec = await embed(`${g.term} ${g.text}`);
        const metaDoc: any = {
          _id: id,
          uid: id,
          type: "glossary",
          key,
          text: g.text,
          display: `${g.term}: ${g.text}`,
          sourceFile: entry.relativePath,
          sectionId: entry.section.id,
          sectionLabel: entry.section.label,
          $vector: vec,
          model: embedModel,
        };
        await collection.replaceOne({ _id: id } as any, metaDoc, { upsert: true } as any);
        metaUpserts++;
      } catch (e: any) {
        console.warn("Failed glossary upsert:", g.term, e?.message || e);
      }
    }

    // ---- meta: ratings items (exact short answers)
    const ratings = extractRatingsItems(docText);
    for (const r of ratings) {
      const key = normKeyText(r.term);
      const id = hashId("rating", `${key}::${r.value}`);
      try {
        const vec = await embed(`rating ${r.term} ${r.value}`);
        const metaDoc: any = {
          _id: id,
          uid: id,
          type: "rating",
          key,
          text: r.value, // store JUST the value so answer is exact
          display: `${r.term}: ${r.value}`,
          sourceFile: entry.relativePath,
          sectionId: entry.section.id,
          sectionLabel: entry.section.label,
          $vector: vec,
          model: embedModel,
        };
        await collection.replaceOne({ _id: id } as any, metaDoc, { upsert: true } as any);
        metaUpserts++;
      } catch (e: any) {
        console.warn("Failed rating upsert:", r.term, e?.message || e);
      }
    }

    // ---- meta: assessments (pre/post test answer-only docs)
    if (entry.section.id === "assessments") {
      const stages: Array<{ stage: AssessmentStage; text: string | null }> = [
        { stage: "pretest", text: extractAssessmentStageText(docText, "pretest") },
        { stage: "posttest", text: extractAssessmentStageText(docText, "posttest") },
      ];

      for (const st of stages) {
        if (!st.text) continue;
        const blocks = extractAssessmentBlocks(st.text);
        for (const b of blocks) {
          const question = extractAssessmentQuestionLine(b.block);
          const answer = extractAssessmentAnswerOnly(b.block) || "";
          const id = hashId("assessment", `${entry.relativePath}::${st.stage}::${b.number}`);

          try {
            const vec = await embed(
              `assessment ${st.stage} question ${b.number} ${question} ${b.block}`.trim()
            );
            const metaDoc: any = {
              _id: id,
              uid: id,
              type: "assessment",
              stage: st.stage,
              number: b.number,
              question,
              answer,
              block: b.block,
              text: answer,
              display: answer,
              sourceFile: entry.relativePath,
              sectionId: entry.section.id,
              sectionLabel: entry.section.label,
              $vector: vec,
              model: embedModel,
            };
            await collection.replaceOne({ _id: id } as any, metaDoc, { upsert: true } as any);
            metaUpserts++;
          } catch (e: any) {
            console.warn(`Failed assessment upsert: ${st.stage} ${b.number}`, e?.message || e);
          }
        }
      }
    }

    // ---- store normal text chunks
    const chunks = await splitter.splitText(docText);
    console.log(`Split "${entry.section.label}" into ${chunks.length} chunks.`);

    for (const raw of chunks) {
      const chunk = normalize(raw);
      if (!chunk || chunk.length < 5) {
        skipped++;
        continue;
      }

      const uid = hashId(`txt_${entry.section.id}`, chunk);

      try {
        const vector = await embed(chunk);

        const doc: any = {
          _id: uid,
          uid,
          type: "text",
          text: chunk,
          model: embedModel,
          sectionId: entry.section.id,
          sectionLabel: entry.section.label,
          sectionDescription: entry.section.description,
          sectionKeywords: entry.section.keywords,
          sourceFile: entry.relativePath,
          $vector: vector,
        };

        await collection.insertOne(doc);
        added++;
      } catch {
        try {
          const vector = await embed(chunk);
          const doc: any = {
            _id: uid,
            uid,
            type: "text",
            text: chunk,
            model: embedModel,
            sectionId: entry.section.id,
            sectionLabel: entry.section.label,
            sectionDescription: entry.section.description,
            sectionKeywords: entry.section.keywords,
            sourceFile: entry.relativePath,
            $vector: vector,
          };
          await collection.replaceOne({ _id: uid } as any, doc, { upsert: true } as any);
          updated++;
        } catch (e2: any) {
          skipped++;
          console.warn("Failed to upsert chunk:", e2?.message || e2);
        }
      }
    }
  }

  console.log(`Meta upserts: ${metaUpserts}`);
  console.log(`Text added: ${added}, updated: ${updated}, skipped: ${skipped}`);
  console.log("All embeddings processed.");
};

// Run
(async () => {
  await createCollection();
  await loadAndStoreEmbeddings();
})();
