import type { ChatStreamEvent } from "@/lib/server/chat/types";

type Writer = {
  send: (event: ChatStreamEvent) => void;
};

export function createNdjsonStream(
  run: (writer: Writer) => Promise<void>
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const writer: Writer = {
        send: (event) => {
          controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
        },
      };

      try {
        await run(writer);
      } finally {
        controller.close();
      }
    },
  });
}
