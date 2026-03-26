import crypto from "crypto";
import { NextRequest, NextResponse } from "next/server";
import { executeChat } from "@/lib/server/chat/engine";
import { persistChatTurn } from "@/lib/server/chat/history";
import { checkRateLimit } from "@/lib/server/chat/rate-limit";
import { createNdjsonStream } from "@/lib/server/chat/stream";
import { parseChatRequest } from "@/lib/server/chat/validate";
import type { ChatStreamEvent } from "@/lib/server/chat/types";

function clientIp(req: NextRequest): string {
  const fromHeader = req.headers.get("x-forwarded-for") || req.headers.get("x-real-ip") || "";
  return fromHeader.split(",")[0]?.trim() || "unknown";
}

function errorPayload(message: string, requestId: string) {
  return {
    error: message,
    requestId,
  };
}

export async function POST(req: NextRequest) {
  const requestId = crypto.randomUUID();

  try {
    const body = await req.json().catch(() => null);
    const parsed = parseChatRequest(body);
    if (!parsed.ok) {
      const message = "error" in parsed ? parsed.error : "Invalid request.";
      return NextResponse.json(errorPayload(message, requestId), { status: 400 });
    }
    const payload = parsed.data;

    const limiterKey = `${clientIp(req)}:${payload.sessionId ?? "anon"}`;
    const rate = await checkRateLimit(limiterKey);
    if (!rate.allowed) {
      return NextResponse.json(
        {
          ...errorPayload("Rate limit exceeded.", requestId),
          retryAfterMs: rate.retryAfterMs,
          limit: rate.limit,
        },
        {
          status: 429,
          headers: {
            "Retry-After": String(Math.max(1, Math.ceil(rate.retryAfterMs / 1000))),
          },
        }
      );
    }

    if (!payload.stream) {
      const result = await executeChat(payload);
      queueMicrotask(() => {
        void persistChatTurn({
          sessionId: payload.sessionId,
          question: payload.question,
          answer: result.answer,
          sources: result.sources,
          pathway: result.pathway,
          similarity: result.similarity,
        });
      });

      return NextResponse.json(
        {
          requestId,
          answer: result.answer,
          sources: result.sources,
          pathway: result.pathway,
          similarity: result.similarity,
          coverage: result.coverage,
        },
        { headers: { "Cache-Control": "no-store" } }
      );
    }

    const stream = createNdjsonStream(async (writer) => {
      try {
        writer.send({
          type: "meta",
          requestId,
          pathway: "pending",
          similarity: 0,
          coverage: 0,
          sources: [],
        });

        const result = await executeChat(payload, {
          onToken: (value) => writer.send({ type: "delta", value }),
        });

        const meta: ChatStreamEvent = {
          type: "meta",
          requestId,
          pathway: result.pathway,
          similarity: result.similarity,
          coverage: result.coverage,
          sources: result.sources,
        };
        writer.send(meta);

        const done: ChatStreamEvent = {
          type: "done",
          requestId,
          pathway: result.pathway,
          similarity: result.similarity,
          coverage: result.coverage,
          sources: result.sources,
        };
        writer.send(done);

        queueMicrotask(() => {
          void persistChatTurn({
            sessionId: payload.sessionId,
            question: payload.question,
            answer: result.answer,
            sources: result.sources,
            pathway: result.pathway,
            similarity: result.similarity,
          });
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unexpected error";
        writer.send({ type: "error", requestId, message });
      }
    });

    return new NextResponse(stream, {
      headers: {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-store",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error";
    return NextResponse.json(errorPayload(message, requestId), { status: 500 });
  }
}

export const runtime = "nodejs";
