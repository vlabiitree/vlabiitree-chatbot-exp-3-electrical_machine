import crypto from "crypto";
import { getChatConfig } from "@/lib/server/chat/config";
import { insertDoc } from "@/lib/server/chat/astra";

type HistoryPayload = {
  sessionId?: string;
  question: string;
  answer: string;
  sources: string[];
  pathway: string;
  similarity: number;
};

export async function persistChatTurn(payload: HistoryPayload): Promise<void> {
  const cfg = getChatConfig();
  if (!cfg.enableChatHistory) return;

  const now = new Date().toISOString();
  const doc = {
    _id: crypto.randomUUID(),
    type: "chat_turn",
    sessionId: payload.sessionId ?? "anonymous",
    question: payload.question,
    answer: payload.answer,
    sources: payload.sources.slice(0, 6),
    pathway: payload.pathway,
    similarity: payload.similarity,
    createdAt: now,
  };

  await insertDoc(cfg.chatHistoryCollection, doc);
}
