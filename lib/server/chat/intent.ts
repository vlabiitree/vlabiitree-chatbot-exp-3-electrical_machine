const STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "how",
  "in",
  "is",
  "it",
  "of",
  "on",
  "or",
  "that",
  "the",
  "this",
  "to",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "with",
]);

export type AssessmentRequest = {
  stage: "pretest" | "posttest";
  number: number;
};

export function normalizeSpace(input: string): string {
  return input.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

export function normalizeKey(input: string): string {
  return (input || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

export function looksLikeExperimentNameQuestion(q: string): boolean {
  return /\b(name of (the )?experiment|experiment name)\b/i.test(q);
}

export function looksLikeAimQuestion(q: string): boolean {
  return /\b(aim|objective(s)?)\b/i.test(q);
}

export function looksLikeTheoryQuestion(q: string): boolean {
  return /\b(theory|principle)\b/i.test(q);
}

export function looksLikeProcedureQuestion(q: string): boolean {
  return /\b(procedure|steps?|perform|conduct|method|how\s+to)\b/i.test(q);
}

export function isImageQuestion(q: string): boolean {
  return /\b(image|photo|figure|diagram|symbol|pic|picture|circuit diagram)\b/i.test(q);
}

export function isRatingQuestion(q: string): boolean {
  return /\b(rating|ratings|rated|range)\b/i.test(q);
}

export function looksLikeDefinitionQuestion(q: string): boolean {
  const s = normalizeSpace(q || "").toLowerCase();
  if (!/^\s*(what is|define|meaning of)\b/.test(s)) return false;

  // Prevent intent collisions for high-priority experiment query types.
  if (looksLikeExperimentNameQuestion(s)) return false;
  if (looksLikeAimQuestion(s)) return false;
  if (looksLikeTheoryQuestion(s)) return false;
  if (looksLikeProcedureQuestion(s)) return false;
  if (isRatingQuestion(s)) return false;

  return true;
}

export function parseAssessmentQuery(question: string): AssessmentRequest | null {
  const s = normalizeSpace(question.toLowerCase());
  const stage = /\bpre[-\s]?test\b/.test(s) ? "pretest" : /\bpost[-\s]?test\b/.test(s) ? "posttest" : null;
  if (!stage) return null;

  const numberMatch =
    s.match(/\bquestion\s*(\d{1,2})\b/) ||
    s.match(/\bq\s*(\d{1,2})\b/) ||
    s.match(/\bno\.?\s*(\d{1,2})\b/) ||
    s.match(/\b(\d{1,2})\b/);

  const number = numberMatch?.[1] ? Number.parseInt(numberMatch[1], 10) : Number.NaN;
  if (!Number.isFinite(number) || number <= 0) return null;
  return { stage, number };
}

export function parseRatingItem(question: string): string | null {
  const s = question.toLowerCase();
  if (/\bvoltmeter\s*2\b|\bdc\s+voltmeter\s*2\b/.test(s)) return "dc voltmeter 2";
  if (/\bvoltmeter\s*1\b|\bdc\s+voltmeter\s*1\b|\bvoltmeter\b/.test(s)) return "dc voltmeter 1";
  if (/\bammeter\s*2\b|\bdc\s+ammeter\s*2\b/.test(s)) return "dc ammeter 2";
  if (/\bammeter\s*1\b|\bdc\s+ammeter\s*1\b|\bammeter\b/.test(s)) return "dc ammeter 1";
  if (/\b(dc\s+shunt\s+motor|shunt\s+motor|dc\s+motor)\b/.test(s)) return "dc motor ratings";
  if (/\bdc\s+shunt\s+generator\b|\bshunt\s+generator\b|\bgenerator\b/.test(s)) return "dc shunt generator";
  if (/\b(3|three)\s*-?\s*point\b/.test(s)) return "3-point starter";
  if (/\blamp\s+load\b/.test(s)) return "lamp load";
  return null;
}

export function parseGlossaryItem(question: string): string | null {
  const s = question.toLowerCase();
  if (/\bmcb\b/.test(s)) return "mcb";
  if (/\b(field\s+rheostat|rheostat)\b/.test(s)) return "field rheostat";
  if (/\b(dc\s+supply|power\s+supply|supply)\b/.test(s)) return "dc supply";
  if (/\b(3|three)\s*-?\s*point\b/.test(s)) return "3-point starter";
  if (/\b(dc\s+shunt\s+motor|shunt\s+motor|dc\s+motor)\b/.test(s)) return "dc motor";
  if (/\bdc\s+shunt\s+generator\b|\bshunt\s+generator\b|\bgenerator\b/.test(s)) return "dc shunt generator";
  if (/\blamp\s+load\b/.test(s)) return "lamp load";
  if (/\bvoltmeter\s*2\b/.test(s)) return "dc voltmeter 2";
  if (/\bvoltmeter\b/.test(s)) return "dc voltmeter 1";
  if (/\bammeter\s*2\b/.test(s)) return "dc ammeter 2";
  if (/\bammeter\b/.test(s)) return "dc ammeter 1";
  return null;
}

export function pickByKeyHint(docs: Record<string, unknown>[], hint: string | null): Record<string, unknown> | null {
  if (!docs.length) return null;
  if (!hint) return docs[0];
  const wanted = normalizeKey(hint);

  for (const doc of docs) {
    const key = normalizeKey(String(doc.key ?? doc.term ?? ""));
    if (!key) continue;
    if (key === wanted || key.includes(wanted) || wanted.includes(key)) return doc;
  }
  return docs[0];
}

function tokenize(input: string): string[] {
  return normalizeSpace(input)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .map((token) => token.trim())
    .filter((token) => token.length >= 3 && !STOPWORDS.has(token));
}

export function lexicalOverlapScore(query: string, text: string): number {
  const queryTokens = Array.from(new Set(tokenize(query)));
  if (!queryTokens.length) return 0;
  const textTokens = new Set(tokenize(text));
  if (!textTokens.size) return 0;

  let matches = 0;
  for (const token of queryTokens) {
    if (textTokens.has(token)) matches += 1;
  }
  return matches / queryTokens.length;
}

export function keywordCoverage(query: string, docs: string[]): number {
  const queryTokens = Array.from(new Set(tokenize(query)));
  if (!queryTokens.length) return 1;
  if (!docs.length) return 0;

  const textTokens = new Set(tokenize(docs.join(" ")));
  let matches = 0;
  for (const token of queryTokens) {
    if (textTokens.has(token)) matches += 1;
  }
  return matches / queryTokens.length;
}
