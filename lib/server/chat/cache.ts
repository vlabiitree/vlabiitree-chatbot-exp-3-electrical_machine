type CacheEntry<T> = {
  value: T;
  expiresAt: number;
};

export class TTLCache<T> {
  private readonly store = new Map<string, CacheEntry<T>>();

  constructor(
    private readonly maxEntries: number,
    private readonly ttlMs: number
  ) {}

  get(key: string): T | null {
    const entry = this.store.get(key);
    if (!entry) return null;

    if (Date.now() >= entry.expiresAt) {
      this.store.delete(key);
      return null;
    }

    // LRU refresh
    this.store.delete(key);
    this.store.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T, ttlOverrideMs?: number): void {
    const ttlMs = Number.isFinite(ttlOverrideMs) && (ttlOverrideMs as number) > 0 ? (ttlOverrideMs as number) : this.ttlMs;
    if (ttlMs <= 0) return;

    while (this.store.size >= this.maxEntries) {
      const oldestKey = this.store.keys().next().value;
      if (!oldestKey) break;
      this.store.delete(oldestKey);
    }

    this.store.set(key, { value, expiresAt: Date.now() + ttlMs });
  }
}
