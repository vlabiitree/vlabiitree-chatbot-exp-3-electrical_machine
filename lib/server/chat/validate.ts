import type { ChatRequest, ClientHistoryMessage } from "@/lib/server/chat/types";

const MAX_QUESTION_CHARS = 1200;
const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_CHARS = 800;

function normalizeSpace(input: string): string {
  return input.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

function stripControl(input: string): string {
  return input.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "");
}

function parseHistory(raw: unknown): ClientHistoryMessage[] {
  if (!Array.isArray(raw)) return [];
  const out: ClientHistoryMessage[] = [];

  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const role = String((item as Record<string, unknown>).role ?? "").trim();
    const content = String((item as Record<string, unknown>).content ?? "");
    if (!["user", "assistant"].includes(role)) continue;
    const cleaned = normalizeSpace(stripControl(content));
    if (!cleaned) continue;
    out.push({
      role: role as "user" | "assistant",
      content: cleaned.slice(0, MAX_HISTORY_CHARS),
    });
    if (out.length >= MAX_HISTORY_MESSAGES) break;
  }
  return out;
}

export function parseChatRequest(body: unknown): { ok: true; data: ChatRequest } | { ok: false; error: string } {
  if (!body || typeof body !== "object") return { ok: false, error: "Invalid JSON body." };

  const record = body as Record<string, unknown>;
  const rawQuestion = String(record.question ?? record.message ?? "");
  const question = normalizeSpace(stripControl(rawQuestion));

  if (!question) return { ok: false, error: "Missing question." };
  if (question.length > MAX_QUESTION_CHARS) {
    return { ok: false, error: `Question is too long. Maximum ${MAX_QUESTION_CHARS} characters.` };
  }

  const history = parseHistory(record.history);
  const stream = record.stream !== false;
  const sessionId = String(record.sessionId ?? "").trim() || undefined;

  return {
    ok: true,
    data: {
      question,
      history,
      stream,
      sessionId,
    },
  };
}
