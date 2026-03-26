import { matchSectionsForQuery } from "@/config/docSections";
import { TTLCache } from "@/lib/server/chat/cache";
import { getChatConfig } from "@/lib/server/chat/config";
import { findDocs, similarityFromDoc, vectorFind } from "@/lib/server/chat/astra";
import { embedQuery } from "@/lib/server/chat/embeddings";
import { NO_ANSWER_MESSAGE } from "@/lib/server/chat/constants";
import {
  isImageQuestion,
  isRatingQuestion,
  keywordCoverage,
  lexicalOverlapScore,
  looksLikeAimQuestion,
  looksLikeDefinitionQuestion,
  looksLikeProcedureQuestion,
  looksLikeExperimentNameQuestion,
  looksLikeTheoryQuestion,
  normalizeSpace,
  parseAssessmentQuery,
  parseGlossaryItem,
  parseRatingItem,
  pickByKeyHint,
} from "@/lib/server/chat/intent";
import type { RetrievedContext } from "@/lib/server/chat/types";

let retrievalCache: TTLCache<RetrievedContext> | null = null;

function cacheKey(question: string): string {
  return normalizeSpace(question).toLowerCase();
}

function safeText(input: unknown): string {
  return typeof input === "string" ? input.trim() : "";
}

function truncate(input: string, maxChars: number): string {
  if (input.length <= maxChars) return input;
  return `${input.slice(0, maxChars)}...`;
}

function dedupeBy<T>(items: T[], toKey: (item: T) => string): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const item of items) {
    const key = toKey(item);
    if (!key) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

function directResult(pathway: string, answer: string, similarity = 1): RetrievedContext {
  const text = normalizeSpace(answer);
  return {
    pathway,
    context: text,
    sources: text ? [text] : [],
    images: [],
    similarity,
    directAnswer: text || NO_ANSWER_MESSAGE,
  };
}

function extractAssessmentAnswer(block: string): string | null {
  const src = normalizeSpace(block || "");
  if (!src) return null;
  const hit = src.match(/\b(correct\s*answer|answer)\s*:\s*(.+)$/i);
  if (!hit?.[2]) return null;
  const raw = hit[2].replace(/\s*(because|reason|explanation)\b[\s\S]*$/i, "").trim();
  if (!raw) return null;
  return raw.replace(/^\(?\s*[a-d]\s*\)?\s*[).:-]?\s*/i, "").trim() || raw;
}

function builtInExperimentDefinition(question: string, experimentContext: string): string | null {
  const q = normalizeSpace(question.toLowerCase());
  if (isRatingQuestion(q) || isImageQuestion(q)) return null;

  const isDefinition = looksLikeDefinitionQuestion(q);
  const compact = q.replace(/[?.!]+$/g, "");
  if (!isDefinition && compact.split(/\s+/).length > 2) return null;

  if (/\belectric current\b|\bcurrent\b/.test(compact)) {
    return `In ${experimentContext}, current is the flow of electric charge and is measured in amperes (A).`;
  }
  if (/\bpotential difference\b|\bvoltage\b/.test(compact)) {
    return `In ${experimentContext}, voltage is the potential difference between two points, measured in volts (V).`;
  }
  if (/\bvoltmeter\b/.test(compact)) {
    return "A DC voltmeter measures terminal or load voltage in volts (V).";
  }
  if (/\bammeter\b/.test(compact)) {
    return "A DC ammeter measures field or load current in amperes (A).";
  }
  return null;
}

function normalizeDirectAnswer(doc: Record<string, unknown>): string {
  return (
    safeText(doc.answer) ||
    safeText(doc.display) ||
    safeText(doc.text) ||
    extractAssessmentAnswer(safeText(doc.block)) ||
    ""
  );
}

export function queryCoverage(question: string, sources: string[]): number {
  return keywordCoverage(question, sources);
}

export async function retrieveContext(question: string): Promise<RetrievedContext> {
  const cfg = getChatConfig();
  if (!retrievalCache) {
    retrievalCache = new TTLCache<RetrievedContext>(cfg.cacheMaxEntries, cfg.retrievalCacheTtlMs);
  }

  const key = cacheKey(question);
  const cached = retrievalCache.get(key);
  if (cached) return cached;

  const wantsExperimentName = looksLikeExperimentNameQuestion(question);
  const wantsAim = looksLikeAimQuestion(question);
  const wantsTheory = looksLikeTheoryQuestion(question);
  const wantsProcedure = looksLikeProcedureQuestion(question);
  const wantsRating = isRatingQuestion(question);
  const glossaryHint = parseGlossaryItem(question);
  const ratingHint = parseRatingItem(question);
  const wantsDefinition =
    looksLikeDefinitionQuestion(question) &&
    !wantsExperimentName &&
    !wantsAim &&
    !wantsTheory &&
    !wantsProcedure &&
    !wantsRating;

  const builtIn = builtInExperimentDefinition(question, cfg.experimentContext);
  if (builtIn) {
    const result = directResult("direct_builtin", builtIn, 1);
    retrievalCache.set(key, result);
    return result;
  }

  const assessment = parseAssessmentQuery(question);
  if (assessment) {
    const exact = await findDocs(
      { type: "assessment", stage: assessment.stage, number: assessment.number },
      { limit: 1, projection: { answer: 1, text: 1, display: 1, block: 1 } }
    );
    const answer = exact.length ? normalizeDirectAnswer(exact[0] as Record<string, unknown>) : "";
    if (answer) {
      const result = directResult("direct_assessment", answer, 1);
      retrievalCache.set(key, result);
      return result;
    }
  }

  if (wantsExperimentName) {
    const nameDoc = await findDocs(
      { type: "meta", key: "experiment_name" },
      { limit: 1, projection: { text: 1, display: 1 } }
    );
    const answer = nameDoc.length ? normalizeDirectAnswer(nameDoc[0] as Record<string, unknown>) : "";
    if (answer) {
      const result = directResult("direct_experiment_name", answer, 1);
      retrievalCache.set(key, result);
      return result;
    }
  }

  const intentSuffix =
    assessment ? `${assessment.stage} question ${assessment.number}` :
    wantsRating ? `ratings ${ratingHint || ""}` :
    wantsAim ? "aim objective of the experiment" :
    wantsTheory ? "theory principle explanation" :
    wantsProcedure ? "procedure steps method" :
    wantsDefinition ? `definition ${glossaryHint || question}` :
    isImageQuestion(question) ? "diagram image symbol figure" :
    "";
  const queryVector = await embedQuery(`${question} ${intentSuffix}`.trim());

  if (wantsRating) {
    const ratingDocs = await vectorFind(
      { type: "rating" },
      queryVector,
      20,
      { projection: { key: 1, text: 1, display: 1 } }
    );
    const picked = pickByKeyHint(ratingDocs as Record<string, unknown>[], ratingHint);
    if (picked) {
      const answer = normalizeDirectAnswer(picked);
      if (answer) {
        const result = directResult("direct_rating", answer, similarityFromDoc(picked));
        retrievalCache.set(key, result);
        return result;
      }
    }
  }

  if (wantsDefinition && glossaryHint) {
    const glossaryDocs = await vectorFind(
      { type: "glossary" },
      queryVector,
      20,
      { projection: { key: 1, text: 1, display: 1 } }
    );
    const picked = pickByKeyHint(glossaryDocs as Record<string, unknown>[], glossaryHint);
    if (picked) {
      const answer = normalizeDirectAnswer(picked);
      if (answer) {
        const result = directResult("direct_glossary", answer, similarityFromDoc(picked));
        retrievalCache.set(key, result);
        return result;
      }
    }
  }

  if (isImageQuestion(question)) {
    const matchedSections = matchSectionsForQuery(question).map((s) => s.id);
    const filters = [...matchedSections.map((sectionId) => ({ type: "image", sectionId })), { type: "image" }];

    const hits: Record<string, unknown>[] = [];
    for (const filter of filters) {
      const docs = await vectorFind(filter, queryVector, 8, { projection: { path: 1 } });
      if (docs.length) hits.push(...(docs as Record<string, unknown>[]));
      if (hits.length >= 8) break;
    }

    const paths = dedupeBy(hits, (doc) => safeText(doc.path)).map((doc) =>
      safeText(doc.path).replace(/\\/g, "/")
    );

    if (paths.length) {
      const markdown = paths
        .slice(0, 6)
        .map((p) => `![](/${String(p).replace(/^\/+/, "")})`)
        .join("\n");
      const similarity = hits.length ? similarityFromDoc(hits[0]) : 0.6;
      const result: RetrievedContext = {
        pathway: "direct_image",
        context: markdown,
        sources: [],
        images: paths.slice(0, 6),
        similarity,
        directAnswer: markdown,
      };
      retrievalCache.set(key, result);
      return result;
    }
  }

  const sections = matchSectionsForQuery(question).map((s) => s.id);
  const sectionLimit = cfg.searchPerSectionLimit;

  const bySection = await Promise.all(
    sections.map(async (sectionId) => {
      const docs = await vectorFind(
        { type: "text", sectionId },
        queryVector,
        sectionLimit,
        { projection: { _id: 1, uid: 1, text: 1, sectionId: 1 } }
      );
      return { sectionId, docs: docs as Record<string, unknown>[] };
    })
  );

  let textDocs = bySection.flatMap((entry) => entry.docs);
  if (!textDocs.length) {
    const fallback = await vectorFind(
      { type: "text" },
      queryVector,
      Math.max(sectionLimit * 2, 10),
      { projection: { _id: 1, uid: 1, text: 1, sectionId: 1 } }
    );
    textDocs = fallback as Record<string, unknown>[];
  }

  textDocs = dedupeBy(textDocs, (doc) => safeText(doc._id) || safeText(doc.uid) || safeText(doc.text));

  const ranked = textDocs
    .map((doc) => {
      const text = safeText(doc.text);
      const sim = similarityFromDoc(doc);
      const lexical = lexicalOverlapScore(question, text);
      const score = sim * 0.78 + lexical * 0.22;
      return { doc, text, sim, score };
    })
    .filter((entry) => Boolean(entry.text))
    .sort((a, b) => b.score - a.score);

  const top = ranked.slice(0, cfg.searchContextDocs);
  const context = top.map((entry) => entry.text).join("\n\n---\n\n");
  const sources = top.map((entry) => truncate(entry.text, 420));
  const similarity = top.length ? top[0].sim : 0;

  const result: RetrievedContext = {
    pathway: "vector_text",
    context,
    sources,
    images: [],
    similarity,
    directAnswer: null,
  };

  retrievalCache.set(key, result);
  return result;
}
