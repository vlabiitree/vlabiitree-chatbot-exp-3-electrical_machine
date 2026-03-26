import type { ClientHistoryMessage } from "@/lib/server/chat/types";
import { normalizeSpace } from "@/lib/server/chat/intent";
import { NO_ANSWER_MESSAGE } from "@/lib/server/chat/constants";

const MAX_HISTORY_TURNS = 6;
const MAX_CONTEXT_CHARS = 3200;

function clamp(input: string, maxChars: number): string {
  if (input.length <= maxChars) return input;
  return `${input.slice(0, maxChars)}...`;
}

function compactHistory(history: ClientHistoryMessage[]): string {
  if (!history.length) return "";
  return history
    .slice(-MAX_HISTORY_TURNS)
    .map((m) => `${m.role === "assistant" ? "Assistant" : "User"}: ${clamp(normalizeSpace(m.content), 280)}`)
    .join("\n");
}

export function buildPrompt(params: {
  question: string;
  context: string;
  history: ClientHistoryMessage[];
}): string {
  const context = clamp(normalizeSpace(params.context), MAX_CONTEXT_CHARS);
  const history = compactHistory(params.history);

  const systemRules = [
    "You are a virtual lab assistant for a Speed Control of DC motor by field resistance control.",
    "Use only the provided context.",
    `If the context does not contain the answer, reply exactly: "${NO_ANSWER_MESSAGE}"`,
    "Keep answers concise and factual.",
    "For procedures, format as a numbered list.",
    "Do not include hidden reasoning, policy text, or unrelated content.",
  ].join("\n");

  return [
    systemRules,
    "",
    "Context:",
    context || "(none)",
    "",
    history ? "Recent conversation:\n" + history + "\n" : "",
    `Question: ${params.question}`,
    "Answer:",
  ]
    .filter(Boolean)
    .join("\n");
}
