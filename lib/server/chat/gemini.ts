import { GoogleGenerativeAI } from "@google/generative-ai";
import { getChatConfig } from "@/lib/server/chat/config";
import { NO_ANSWER_MESSAGE } from "@/lib/server/chat/constants";
import { buildPrompt } from "@/lib/server/chat/prompt";
import type { ClientHistoryMessage } from "@/lib/server/chat/types";

type StreamParams = {
  question: string;
  context: string;
  history: ClientHistoryMessage[];
  onToken?: (token: string) => void;
};

let modelPromise: Promise<ReturnType<GoogleGenerativeAI["getGenerativeModel"]>> | null = null;

async function getModel() {
  if (!modelPromise) {
    modelPromise = Promise.resolve().then(() => {
      const cfg = getChatConfig();
      const client = new GoogleGenerativeAI(cfg.geminiApiKey);
      return client.getGenerativeModel(
        {
          model: cfg.geminiModel,
          generationConfig: {
            temperature: cfg.geminiTemperature,
            maxOutputTokens: cfg.geminiMaxOutputTokens,
          },
        },
        { apiVersion: cfg.geminiApiVersion }
      );
    });
  }
  return modelPromise;
}

function extractChunkText(chunk: unknown): string {
  if (!chunk) return "";

  type GeminiChunk = {
    text?: () => string;
    candidates?: Array<{
      content?: {
        parts?: Array<{ text?: string }>;
      };
    }>;
  };

  const typed = chunk as GeminiChunk;

  if (typeof typed.text === "function") {
    try {
      return String(typed.text() ?? "");
    } catch {
      return "";
    }
  }

  const candidates = typed.candidates;
  if (!Array.isArray(candidates)) return "";
  return candidates
    .flatMap((c) => c.content?.parts ?? [])
    .map((p) => String(p?.text ?? ""))
    .join("");
}

export async function streamGeminiAnswer(params: StreamParams): Promise<string> {
  const model = await getModel();
  const prompt = buildPrompt({
    question: params.question,
    context: params.context,
    history: params.history,
  });

  const result = await model.generateContentStream({
    contents: [{ role: "user", parts: [{ text: prompt }] }],
  });

  let output = "";
  for await (const chunk of result.stream) {
    const delta = extractChunkText(chunk);
    if (!delta) continue;
    output += delta;
    params.onToken?.(delta);
  }

  const normalized = output.trim();
  if (normalized) return normalized;

  try {
    const fallback = await result.response;
    const text = String(fallback.text?.() ?? "").trim();
    return text || NO_ANSWER_MESSAGE;
  } catch {
    return NO_ANSWER_MESSAGE;
  }
}
