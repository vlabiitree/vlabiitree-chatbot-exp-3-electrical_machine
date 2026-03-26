export type ChatRole = "user" | "assistant";

export type ClientHistoryMessage = {
  role: ChatRole;
  content: string;
};

export type ChatRequest = {
  question: string;
  history: ClientHistoryMessage[];
  stream: boolean;
  sessionId?: string;
};

export type RetrievedContext = {
  pathway: string;
  context: string;
  sources: string[];
  images: string[];
  similarity: number;
  directAnswer: string | null;
};

export type ChatExecution = {
  answer: string;
  pathway: string;
  sources: string[];
  similarity: number;
  coverage: number;
};

export type ChatStreamEvent =
  | {
      type: "meta";
      requestId: string;
      pathway: string;
      similarity: number;
      coverage: number;
      sources: string[];
    }
  | { type: "delta"; value: string }
  | {
      type: "done";
      requestId: string;
      pathway: string;
      similarity: number;
      coverage: number;
      sources: string[];
    }
  | { type: "error"; requestId: string; message: string };

export type RateLimitState = {
  allowed: boolean;
  limit: number;
  remaining: number;
  retryAfterMs: number;
  source: "memory" | "upstash";
};
