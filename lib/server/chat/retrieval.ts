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
  const raw = hit[2].trim();
  if (!raw) return null;
  return raw.replace(/^\(?\s*[a-d]\s*\)?\s*[).:-]\s*/i, "").trim() || raw;
}

function escapeRegExp(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isolateAssessmentStageText(input: string, stage: "pretest" | "posttest"): string {
  const text = input || "";
  if (!text) return "";

  const preIdx = text.search(/\bpre\s*-?\s*test\b/i);
  const postIdx = text.search(/\bpost\s*-?\s*test\b/i);

  if (stage === "pretest") {
    if (preIdx >= 0 && postIdx > preIdx) return text.slice(preIdx, postIdx);
    if (preIdx >= 0) return text.slice(preIdx);
    // If the chunk has only posttest marker, ignore it for pretest lookups.
    if (postIdx >= 0) return "";
    return text;
  }

  if (postIdx >= 0) return text.slice(postIdx);
  // If the chunk has only pretest marker, ignore it for posttest lookups.
  if (preIdx >= 0) return "";
  return text;
}

function extractAssessmentAnswerFromText(
  input: string,
  stage: "pretest" | "posttest",
  number: number
): string | null {
  const scoped = normalizeSpace(isolateAssessmentStageText(input, stage));
  if (!scoped) return null;

  const n = escapeRegExp(String(number));
  const blockPattern = new RegExp(
    `(?:^|\\n)\\s*${n}[.)]\\s+([\\s\\S]*?)(?=(?:\\n\\s*\\d{1,2}[.)]\\s+)|$)`,
    "i"
  );
  const blockHit = scoped.match(blockPattern);
  if (blockHit?.[0]) {
    const answer = extractAssessmentAnswer(blockHit[0]);
    if (answer) return answer;
  }

  const inlinePattern = new RegExp(
    `\\b${n}[.)][^\\n]{0,320}(?:correct\\s*answer|answer)\\s*:\\s*([^\\n]+)`,
    "i"
  );
  const inlineHit = scoped.match(inlinePattern);
  if (!inlineHit) return null;

  const parsedInline = extractAssessmentAnswer(inlineHit[0]);
  if (parsedInline) return parsedInline;

  const fallback = inlineHit[1]?.trim();
  if (!fallback) return null;
  return fallback.replace(/^\(?\s*[a-d]\s*\)?\s*[).:-]\s*/i, "").trim() || fallback;
}

function normalizeExperimentNameCandidate(input: string): string | null {
  const cleaned = normalizeSpace(input || "")
    .replace(/^[\s"'“”]+|[\s"'“”]+$/g, "")
    .replace(/\s+/g, " ")
    .replace(/^the\s+/i, "")
    .replace(/\s*\.\s*$/, "")
    .trim();
  if (!cleaned) return null;
  const generic = cleaned.toLowerCase();
  if (generic === "this experiment" || generic === "the experiment" || generic === "current experiment") {
    return null;
  }
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

function deriveExperimentNameFromText(input: string): string | null {
  const text = normalizeSpace(input || "");
  if (!text) return null;

  const explicitPatterns = [
    /name of the experiment\s*(?:is)?\s*[:\-]\s*([^\n.]+)/i,
    /experiment name\s*[:\-]\s*([^\n.]+)/i,
    /welcome to (?:the )?experiment\s*["“]?([^"”\n.]+)["”]?/i,
  ];
  for (const pattern of explicitPatterns) {
    const hit = text.match(pattern);
    if (hit?.[1]) {
      const value = normalizeExperimentNameCandidate(hit[1]);
      if (value) return value;
    }
  }

  const aimBased = text.match(/\bto study\s+([^\n.]+?)(?:,\s*and\s+plot[\s\S]*?)?(?:\.|$)/i);
  if (aimBased?.[1]) {
    return normalizeExperimentNameCandidate(aimBased[1]);
  }
  return null;
}

function titleCase(input: string): string {
  return normalizeSpace(input)
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function splitDefinitionUnits(input: string): string[] {
  return normalizeSpace(input)
    .split(/\n+|(?<=[.!?])\s+|(?<=:)\s+/)
    .map((part) => normalizeSpace(part))
    .filter(Boolean);
}

function definitionNoisePenalty(input: string): number {
  const text = input.toLowerCase();
  let penalty = 0;
  if ((text.match(/\banswer\s*:/g) || []).length >= 2) penalty += 1.2;
  if (/reading added|add to table|duplicate reading|click add|audio|alert/.test(text)) penalty += 0.8;
  if (/\b\d+\s*(st|nd|rd|th)\s*time\b/.test(text)) penalty += 0.9;
  if (/\b\d+(\.\d+)?\s*(a|amp|amps|rpm|volt|volts)\b/.test(text)) penalty += 0.85;
  return penalty;
}

function deriveDefinitionByHintFromDocs(
  hint: string,
  docs: Record<string, unknown>[]
): string | null {
  const hintNorm = normalizeSpace(hint).toLowerCase();
  if (!hintNorm || !docs.length) return null;

  const all = docs
    .map((doc) => safeText(doc.text))
    .filter(Boolean)
    .join("\n")
    .toLowerCase();

  if (/\brheostat\b/.test(hintNorm)) {
    if (
      /\b(field\s+rheostat|rheostat)\b/.test(all) &&
      /(vary the field resistance|vary the field current|controls?\s+the field current|move(?:ing)? the rheostat knob)/.test(all)
    ) {
      return "Field rheostat is a variable resistor used to vary the field current.";
    }
    if (/\barmature rheostat\b/.test(all)) {
      return "Armature rheostat is used to vary armature resistance/current for control and observations.";
    }
  }

  if (/\bspeed\b/.test(hintNorm)) {
    if (/\bn\s*=\s*rotational speed of the motor\b|\brotational speed of the motor\b/.test(all)) {
      return "Speed (N) is the rotational speed of the motor, measured in rpm.";
    }
  }

  return null;
}

type ProcedureTemplate = {
  id: string;
  pattern: RegExp;
  step: string;
};

const PROCEDURE_TEMPLATES: ProcedureTemplate[] = [
  {
    id: "open",
    pattern: /\bcomponents window\b|\bstart the simulation\b/i,
    step: "Open the simulation and review the components.",
  },
  {
    id: "connect",
    pattern: /\bautoconnect completed\b|\bclick on the check button\b|\bcorrect connections\b|\bconnections:\s*let.?s connect\b/i,
    step: "Complete the circuit connections and click Check to verify them.",
  },
  {
    id: "supply",
    pattern: /\bdc supply has been turned on\b|\bdc supply on\b/i,
    step: "Turn ON the DC supply.",
  },
  {
    id: "starter",
    pattern: /\bstarter handle from left to right\b/i,
    step: "Move the starter handle from left to right.",
  },
  {
    id: "armature",
    pattern: /\bset the armature rheostat\b|\barmature resistance is set\b/i,
    step: "Set the armature rheostat to the initial condition.",
  },
  {
    id: "vary_field",
    pattern: /\bvary the field resistance\b|\bfield rheostat variation\b|\bmoving the rheostat knob\b/i,
    step: "Vary the field rheostat to change field current and speed.",
  },
  {
    id: "record",
    pattern: /\badd to table\b|\bobservation table\b|\breading added\b/i,
    step: "After each setting, click Add to Table to record field current and speed.",
  },
  {
    id: "repeat",
    pattern: /\bnext reading\b|\bmaximum of 7 readings\b|\ball 7 readings\b/i,
    step: "Repeat variation and recording for successive readings (up to 7).",
  },
  {
    id: "graph",
    pattern: /\bgraph button\b|\bplot the graph\b|\bgraph of speed vs field current\b/i,
    step: "Click Graph to plot the Speed vs Field Current curve.",
  },
  {
    id: "report",
    pattern: /\breport button\b|\bgenerate your report\b|\bprint to print\b|\breset to start again\b/i,
    step: "Generate the report, then print or reset if required.",
  },
];

function procedureNoise(input: string): number {
  const text = input.toLowerCase();
  let penalty = 0;
  if (/wrong connection|duplicate reading|some connections are wrong/.test(text)) penalty += 1;
  if (/audio|alert/.test(text)) penalty += 0.6;
  if (/\b\d+(\.\d+)?\s*(a|rpm|volt|volts)\b/.test(text)) penalty += 0.8;
  return penalty;
}

function dedupeStrings(input: string[]): string[] {
  return dedupeBy(input, (item) => normalizeSpace(item).toLowerCase());
}

function parseProcedureFromDocs(docs: Record<string, unknown>[]): string | null {
  if (!docs.length) return null;
  const raw = docs.map((doc) => safeText(doc.text)).filter(Boolean).join("\n");
  const text = normalizeSpace(raw);
  if (!text) return null;

  const steps: string[] = [];
  for (const template of PROCEDURE_TEMPLATES) {
    if (template.pattern.test(text)) steps.push(template.step);
  }

  if (steps.length < 5) {
    const sentences = text
      .split(/(?<=[.!?])\s+|\n+/)
      .map((sentence) => normalizeSpace(sentence))
      .filter(Boolean);

    const verbs = /\b(connect|turn on|move|set|vary|click|add|record|repeat|plot|generate|print|reset)\b/i;
    const fallbackSteps = sentences
      .map((sentence) => {
        const score =
          lexicalOverlapScore("procedure steps experiment", sentence) +
          (verbs.test(sentence) ? 0.8 : 0) -
          procedureNoise(sentence);
        return { sentence, score };
      })
      .filter((item) => item.score >= 0.55)
      .sort((a, b) => b.score - a.score)
      .map((item) => item.sentence)
      .slice(0, 8);

    steps.push(...fallbackSteps);
  }

  const clean = dedupeStrings(steps)
    .map((step) => step.replace(/\s*-\s*(audio|alert)\b/gi, "").trim())
    .filter(Boolean)
    .slice(0, 10);

  if (!clean.length) return null;
  return clean.map((step, index) => `${index + 1}. ${step}`).join("\n");
}

function extractDefinitionFromText(input: string, hint: string): string | null {
  const hintNorm = normalizeSpace(hint).toLowerCase();
  if (!hintNorm) return null;

  const tokens = hintNorm.split(/\s+/).filter(Boolean);
  const units = splitDefinitionUnits(input);
  if (!units.length) return null;

  const scored = units
    .map((unit) => {
      const normalized = unit.toLowerCase();
      const tokenMatches = tokens.filter((token) => normalized.includes(token)).length;
      const exactBonus = normalized.includes(hintNorm) ? 1.4 : 0;
      const defBonus = /\b(is|means|refers to|used to|controls?|measures?)\b/.test(normalized) ? 0.4 : 0;
      const score =
        lexicalOverlapScore(`${hintNorm} definition`, unit) +
        tokenMatches / Math.max(tokens.length, 1) +
        exactBonus +
        defBonus -
        definitionNoisePenalty(unit);
      return { unit, score };
    })
    .sort((a, b) => b.score - a.score);

  const best = scored[0];
  if (!best || best.score < 0.6) return null;

  const cleaned = best.unit.replace(/\s*-\s*(audio|alert)\b/gi, "").trim();
  if (!cleaned) return null;
  return cleaned.length > 240 ? `${cleaned.slice(0, 237)}...` : cleaned;
}

function findDefinitionFromAssessments(
  docs: Record<string, unknown>[],
  hint: string
): string | null {
  const hintNorm = normalizeSpace(hint).toLowerCase();
  if (!hintNorm || !docs.length) return null;
  const tokens = hintNorm.split(/\s+/).filter(Boolean);

  const scored = docs
    .map((doc) => {
      const question = normalizeSpace(safeText(doc.question));
      const answer = normalizeSpace(normalizeDirectAnswer(doc));
      const qLower = question.toLowerCase();
      const aLower = answer.toLowerCase();
      const tokenMatches =
        tokens.filter((token) => qLower.includes(token) || aLower.includes(token)).length /
        Math.max(tokens.length, 1);
      const usedToBonus = /\b(used to|controls?|component|purpose)\b/i.test(question) ? 0.45 : 0;
      const exactBonus = qLower.includes(hintNorm) ? 1 : 0;
      const score =
        lexicalOverlapScore(`${hintNorm} definition`, `${question} ${answer}`) +
        tokenMatches +
        usedToBonus +
        exactBonus;
      return { question, answer, score };
    })
    .sort((a, b) => b.score - a.score);

  const top = scored[0];
  if (!top || top.score < 1.05 || !top.answer) return null;

  const answerLower = top.answer.toLowerCase();
  if (answerLower.includes(hintNorm)) return top.answer;
  return `${titleCase(hintNorm)} is used to ${top.answer.charAt(0).toLowerCase()}${top.answer.slice(1)}.`;
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

    const fromContext = normalizeExperimentNameCandidate(cfg.experimentContext);
    if (fromContext) {
      const result = directResult("direct_experiment_context", fromContext, 1);
      retrievalCache.set(key, result);
      return result;
    }

    const aimDocs = await findDocs(
      { type: "text", sectionId: "aim_theory_procedure" },
      { limit: 12, projection: { text: 1 } }
    );
    for (const doc of aimDocs as Record<string, unknown>[]) {
      const name = deriveExperimentNameFromText(safeText(doc.text));
      if (!name) continue;
      const result = directResult("direct_experiment_derived", name, 0.9);
      retrievalCache.set(key, result);
      return result;
    }

    const legacyDocs = await findDocs(
      { type: "text", sectionId: "legacy_manual" },
      { limit: 16, projection: { text: 1 } }
    );
    for (const doc of legacyDocs as Record<string, unknown>[]) {
      const name = deriveExperimentNameFromText(safeText(doc.text));
      if (!name) continue;
      const result = directResult("direct_experiment_derived_legacy", name, 0.82);
      retrievalCache.set(key, result);
      return result;
    }
  }

  if (wantsProcedure) {
    const simulationDocs = await findDocs(
      { type: "text", sectionId: "simulation" },
      { limit: 80, projection: { text: 1 } }
    );
    let procedure = parseProcedureFromDocs(simulationDocs as Record<string, unknown>[]);

    if (!procedure) {
      const legacyDocs = await findDocs(
        { type: "text", sectionId: "legacy_manual" },
        { limit: 100, projection: { text: 1 } }
      );
      procedure = parseProcedureFromDocs(legacyDocs as Record<string, unknown>[]);
    }

    if (procedure) {
      const result = directResult("direct_procedure", procedure, 0.92);
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

  if (assessment) {
    const textCandidates = await vectorFind(
      { type: "text" },
      queryVector,
      120,
      { projection: { text: 1 } }
    );
    const merged = (textCandidates as Record<string, unknown>[])
      .map((doc) => safeText(doc.text))
      .filter(Boolean)
      .join("\n");
    const parsed = extractAssessmentAnswerFromText(merged, assessment.stage, assessment.number);
    if (parsed) {
      const similarity = textCandidates.length
        ? similarityFromDoc(textCandidates[0] as Record<string, unknown>)
        : 0.68;
      const result = directResult("direct_assessment_fallback_text", parsed, similarity);
      retrievalCache.set(key, result);
      return result;
    }
  }

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

    const assessmentDocs = await findDocs(
      { type: "assessment" },
      { limit: 160, projection: { question: 1, answer: 1, text: 1, block: 1 } }
    );
    const fromAssessment = findDefinitionFromAssessments(
      assessmentDocs as Record<string, unknown>[],
      glossaryHint
    );
    if (fromAssessment) {
      const result = directResult("direct_definition_assessment", fromAssessment, 0.86);
      retrievalCache.set(key, result);
      return result;
    }

    const allTextForHint = await findDocs(
      { type: "text" },
      { limit: 240, projection: { text: 1 } }
    );
    const patternDerived = deriveDefinitionByHintFromDocs(
      glossaryHint,
      allTextForHint as Record<string, unknown>[]
    );
    if (patternDerived) {
      const result = directResult("direct_definition_pattern", patternDerived, 0.84);
      retrievalCache.set(key, result);
      return result;
    }

    const definitionDocs = await vectorFind(
      { type: "text" },
      queryVector,
      28,
      { projection: { text: 1 } }
    );
    const extracted = (definitionDocs as Record<string, unknown>[])
      .map((doc) => extractDefinitionFromText(safeText(doc.text), glossaryHint))
      .filter(Boolean) as string[];
    if (extracted.length) {
      const result = directResult(
        "direct_definition_text",
        dedupeBy(extracted, (item) => normalizeSpace(item).toLowerCase())[0],
        similarityFromDoc((definitionDocs[0] as Record<string, unknown>) || {})
      );
      retrievalCache.set(key, result);
      return result;
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
