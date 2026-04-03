import { computeScore } from '../internal/metrics';
import type { SimilarityMetric, StoredItem } from '../types';

/**
 * Stateful search index with CRUD operations and brute-force search.
 *
 * Maintains an in-memory corpus of embeddings with string IDs and optional
 * metadata. Suitable for corpora up to ~100k embeddings (linear scan).
 *
 * @example
 * const index = new SearchIndex({ metric: 'cosine' });
 * index.add('doc1', embedding1, { title: 'Hello' });
 * const results = index.search(queryEmbedding, { topK: 5 });
 */
export class SearchIndex {
  private items = new Map<string, StoredItem>();
  private metric: SimilarityMetric;

  constructor(options?: { metric?: SimilarityMetric }) {
    this.metric = options?.metric ?? 'cosine';
  }

  /** Add a single item. Overwrites if ID already exists. */
  add(id: string, embedding: number[], metadata?: Record<string, unknown>): void {
    this.items.set(id, { id, embedding, ...(metadata !== undefined ? { metadata } : {}) });
  }

  /** Add multiple items at once. */
  addBatch(items: Array<{ id: string; embedding: number[]; metadata?: Record<string, unknown> }>): void {
    for (const item of items) {
      this.add(item.id, item.embedding, item.metadata);
    }
  }

  /** Remove an item by ID. Returns true if found and removed. */
  remove(id: string): boolean {
    return this.items.delete(id);
  }

  /** Search for the most similar items to a query vector. */
  search(
    query: number[],
    options?: {
      topK?: number;
      threshold?: number;
      filter?: (item: StoredItem) => boolean;
    },
  ): Array<StoredItem & { score: number }> {
    const topK = options?.topK ?? 10;
    const threshold = options?.threshold;
    const filter = options?.filter;

    const results: Array<StoredItem & { score: number }> = [];

    for (const item of this.items.values()) {
      if (filter && !filter(item)) continue;

      const score = computeScore(query, item.embedding, this.metric);

      if (threshold !== undefined && score < threshold) continue;

      results.push({ ...item, score });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /** Get an item by ID. */
  get(id: string): StoredItem | undefined {
    return this.items.get(id);
  }

  /** Number of items in the index. */
  get size(): number {
    return this.items.size;
  }

  /** Remove all items. */
  clear(): void {
    this.items.clear();
  }
}
