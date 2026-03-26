"use client";

import React from "react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

type ChatApiResponse = {
  answer?: string;
  sources?: string[];
  error?: string;
};

type ChatStreamEvent =
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

const SUGGESTED: string[] = [
  "What is the name of the experiment?",
  "What is the aim of this experiment?",
  "What is the procedure of this experiment?",
];

function makeId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function Home() {
  const normalizeForImages = (text: string): string => {
    if (!text) return "";
    let out = text.replace(
      /\b(Photo|Symbol|Image|Figure|Pic|Picture)\s*:\s*(\/?images\/[^\n\r]+?\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif|svg))/gi,
      (_m, lbl, rawPath) => {
        const base = String(rawPath).startsWith("/") ? String(rawPath) : `/${String(rawPath)}`;
        return `${lbl}: ![](${encodeURI(base)})`;
      }
    );
    out = out.replace(
      /(^|[\s(])(\/?images\/[^\n\r]+?\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif|svg))(?=[\s)\]\}.,!?;:]|$)/gi,
      (_m, lead, rawPath) => {
        const base = String(rawPath).startsWith("/") ? String(rawPath) : `/${String(rawPath)}`;
        return `${lead}![](${encodeURI(base)})`;
      }
    );
    return out;
  };

  const isPureImageMarkdown = (text: string): boolean => {
    if (!text) return false;
    const trimmed = text.trim();
    if (!trimmed) return false;
    const lines = trimmed.split(/\n+/);
    const imgLine = /^!\[\]\(\/?images\/[^\n\r]+?\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif|svg)\)$/i;
    return lines.every((ln) => imgLine.test(ln.trim()));
  };

  const extractImagePathsFromMarkdown = (text: string): string[] => {
    const paths: string[] = [];
    const re = /!\[\]\((\/?images\/[^\n\r]+?\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif|svg))\)/gi;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) paths.push(m[1]);
    return paths;
  };

  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [streamingStarted, setStreamingStarted] = React.useState(false);
  const listRef = React.useRef<HTMLDivElement | null>(null);
  const sessionIdRef = React.useRef<string>(makeId());

  React.useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, loading]);

  const ask = async (question: string) => {
    if (!question.trim() || loading) return;

    const userId = makeId();
    const assistantId = makeId();
    const history = messages.slice(-8).map((m) => ({ role: m.role, content: m.content }));

    setLoading(true);
    setStreamingStarted(false);
    setMessages((existing) => [...existing, { id: userId, role: "user", content: question }]);
    setInput("");

    const updateAssistant = (updater: (prev: Message) => Message) => {
      setMessages((existing) => {
        let found = false;
        const next = existing.map((msg) => {
          if (msg.id !== assistantId) return msg;
          found = true;
          return updater(msg);
        });

        if (found) return next;

        return [
          ...next,
          updater({
            id: assistantId,
            role: "assistant",
            content: "",
            sources: [],
          }),
        ];
      });
    };

    const updateAssistantIfExists = (updater: (prev: Message) => Message) => {
      setMessages((existing) => {
        let found = false;
        const next = existing.map((msg) => {
          if (msg.id !== assistantId) return msg;
          found = true;
          return updater(msg);
        });
        return found ? next : existing;
      });
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history,
          stream: true,
          sessionId: sessionIdRef.current,
        }),
      });

      if (!response.ok) {
        const fallback = (await response.json().catch(() => ({}))) as ChatApiResponse;
        const msg = fallback?.error || `Error ${response.status}`;
        updateAssistant((prev) => ({ ...prev, content: msg }));
        return;
      }

      const contentType = response.headers.get("content-type") || "";
      if (!response.body || !contentType.includes("application/x-ndjson")) {
        const json = (await response.json().catch(() => ({}))) as ChatApiResponse;
        updateAssistant((prev) => ({
          ...prev,
          content: json?.answer || "No answer returned.",
          sources: Array.isArray(json?.sources) ? json.sources : [],
        }));
        setStreamingStarted(true);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let pending = "";
      let flushTimer: ReturnType<typeof setTimeout> | null = null;

      const flush = () => {
        if (!pending) return;
        const chunk = pending;
        pending = "";
        updateAssistant((prev) => ({ ...prev, content: `${prev.content}${chunk}` }));
      };

      const enqueueChunk = (chunk: string) => {
        if (!chunk) return;
        if (!streamingStarted) setStreamingStarted(true);
        pending += chunk;
        if (!flushTimer) {
          flushTimer = setTimeout(() => {
            flushTimer = null;
            flush();
          }, 25);
        }
      };

      const handleEvent = (event: ChatStreamEvent) => {
        if (event.type === "delta") {
          enqueueChunk(event.value);
          return;
        }

        if (event.type === "meta" || event.type === "done") {
          updateAssistantIfExists((prev) => ({
            ...prev,
            sources: Array.isArray(event.sources) ? event.sources : prev.sources,
          }));
          return;
        }

        if (event.type === "error") {
          updateAssistant((prev) => ({
            ...prev,
            content: prev.content ? `${prev.content}\n${event.message}` : event.message,
          }));
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let lineBreak = buffer.indexOf("\n");
        while (lineBreak >= 0) {
          const line = buffer.slice(0, lineBreak).trim();
          buffer = buffer.slice(lineBreak + 1);
          if (line) {
            try {
              handleEvent(JSON.parse(line) as ChatStreamEvent);
            } catch {
              // ignore malformed lines
            }
          }
          lineBreak = buffer.indexOf("\n");
        }
      }

      if (buffer.trim()) {
        try {
          handleEvent(JSON.parse(buffer.trim()) as ChatStreamEvent);
        } catch {
          // ignore malformed tail line
        }
      }

      if (flushTimer) {
        clearTimeout(flushTimer);
        flushTimer = null;
      }
      flush();
    } catch (error) {
      const message = error instanceof Error ? error.message : "request failed";
      updateAssistant((prev) => ({ ...prev, content: `Error: ${message}` }));
    } finally {
      setLoading(false);
      setStreamingStarted(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
      <section className="w-full max-w-4xl h-[min(80vh,720px)] flex flex-col overflow-hidden rounded-xl border border-gray-300 bg-white shadow-lg">
        <div
          ref={listRef}
          className="flex-1 space-y-4 overflow-y-auto px-3 py-3 sm:px-6 sm:py-4 scroll-smooth"
        >
          {messages.length === 0 && !loading && (
            <div className="grid h-full place-items-center">
              <div className="text-center space-y-4 max-w-md">
                <div className="mx-auto h-20 w-20 rounded-full bg-cyan-100 flex items-center justify-center shadow-md">
                  <svg
                    className="h-10 w-10"
                    style={{ color: "#02263C" }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <div className="space-y-3">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">
                      Welcome to Virtual Lab Assistant
                    </h3>
                    <p className="text-sm text-gray-600 mt-2">
                      You can start by asking any of these common questions:
                    </p>
                  </div>
                  <div className="flex flex-col items-stretch gap-2">
                    {SUGGESTED.map((q) => (
                      <button
                        key={q}
                        onClick={() => ask(q)}
                        className="w-full rounded-md bg-cyan-50 hover:bg-cyan-100 border border-cyan-200 px-3 py-2 text-xs font-medium text-[#02263C] transition-all duration-200 hover:shadow-sm text-left"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-200 ${
                m.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {m.role === "assistant" && (
                <div
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-xs font-bold text-white shadow-md"
                  style={{ backgroundColor: "#02263C" }}
                >
                  VL
                </div>
              )}
              <div
                className={
                  "max-w-[78%] rounded-lg px-4 py-3 text-sm shadow-sm transition-all duration-200 " +
                  (m.role === "user"
                    ? "text-white"
                    : "bg-gray-50 text-gray-800 border border-gray-200")
                }
                style={m.role === "user" ? { backgroundColor: "#02263C" } : {}}
              >
                {m.role === "assistant" ? (
                  <>
                    {isPureImageMarkdown(m.content) ? (
                      extractImagePathsFromMarkdown(m.content).map((p, imgIdx) => {
                        const base = p.startsWith("/") ? p : `/${p}`;
                        const src = encodeURI(base);
                        return (
                          <Image
                            key={imgIdx}
                            src={src}
                            alt=""
                            width={1200}
                            height={800}
                            unoptimized
                            className="mt-1 rounded border border-gray-200"
                            style={{
                              maxWidth: "100%",
                              height: "auto",
                            }}
                          />
                        );
                      })
                    ) : (
                      <>
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            img: (props) => {
                              const src = typeof props.src === "string" ? props.src : "";
                              if (!src) return null;
                              return (
                                <Image
                                  src={src}
                                  alt={props.alt ?? ""}
                                  width={1200}
                                  height={800}
                                  unoptimized
                                  style={{
                                    maxWidth: "100%",
                                    height: "auto",
                                    borderRadius: 6,
                                  }}
                                />
                              );
                            },
                            a: (props) => (
                              <a
                                {...props}
                                target="_blank"
                                rel="noreferrer"
                                className="underline"
                              />
                            ),
                          }}
                        >
                          {normalizeForImages(m.content)}
                        </ReactMarkdown>
                        {Array.isArray(m.sources) &&
                          m.sources.some((s) => /(^|\/)images\//.test(String(s))) && (
                            <div className="mt-2 grid grid-cols-2 gap-2">
                              {m.sources
                                .filter((s) => /(^|\/)images\//.test(String(s)))
                                .slice(0, 6)
                                .map((s, i) => {
                                  const raw = String(s).startsWith("/") ? String(s) : `/${String(s)}`;
                                  const src = encodeURI(raw);
                                  return (
                                    <Image
                                      key={i}
                                      src={src}
                                      alt=""
                                      width={1200}
                                      height={800}
                                      unoptimized
                                      className="rounded border border-gray-200"
                                      style={{
                                        maxWidth: "100%",
                                        height: "auto",
                                      }}
                                    />
                                  );
                                })}
                            </div>
                          )}
                      </>
                    )}
                  </>
                ) : (
                  <span className="whitespace-pre-wrap">{m.content}</span>
                )}
              </div>
              {m.role === "user" && (
                <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-gray-600 text-xs font-bold text-white shadow-md">
                  You
                </div>
              )}
            </div>
          ))}

          {loading && !streamingStarted && (
            <div className="flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <div
                className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-xs font-bold text-white shadow-md"
                style={{ backgroundColor: "#02263C" }}
              >
                VL
              </div>
              <div className="max-w-[78%] rounded-lg bg-gray-50 border border-gray-200 px-4 py-3 text-sm text-gray-700 shadow-sm">
                <span className="inline-flex items-center gap-1.5">
                  Processing
                  <span
                    className="inline-block h-1.5 w-1.5 animate-bounce rounded-full [animation-delay:-0.3s]"
                    style={{ backgroundColor: "#02263C" }}
                  ></span>
                  <span
                    className="inline-block h-1.5 w-1.5 animate-bounce rounded-full [animation-delay:-0.15s]"
                    style={{ backgroundColor: "#02263C" }}
                  ></span>
                  <span
                    className="inline-block h-1.5 w-1.5 animate-bounce rounded-full"
                    style={{ backgroundColor: "#02263C" }}
                  ></span>
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="border-t border-gray-200 bg-gray-50 px-3 py-3 sm:px-4 sm:py-3">
          <div
            className="flex items-center gap-3 rounded-lg border border-gray-300 bg-white px-2 py-2 shadow-md transition-all duration-200 focus-within:ring-2 focus-within:border-cyan-500"
            style={{ "--tw-ring-color": "#02263C" } as React.CSSProperties}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              placeholder="Type your question here..."
              className="w-full rounded-md border-0 bg-transparent px-2.5 py-2 text-sm text-gray-800 placeholder:text-gray-500 focus:outline-none focus:ring-0"
            />
            <button
              onClick={() => ask(input)}
              disabled={loading || !input.trim()}
              className="grid h-10 w-10 shrink-0 place-items-center rounded-md text-white shadow-sm transition-all duration-200 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ backgroundColor: "#02263C" }}
              title="Send message"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-5 w-5"
              >
                <path d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}
