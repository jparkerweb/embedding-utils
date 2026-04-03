import { describe, it, expect } from 'vitest';
import { clusterEmbeddings } from '../../src/clustering/cluster';
import { topK } from '../../src/search/topk';
import type { EmbeddingProvider, EmbeddingResult } from '../../src/types';

/**
 * Mock provider that produces deterministic embeddings.
 * Groups texts into "topics" based on their prefix.
 */
function createMockProvider(): EmbeddingProvider {
  return {
    name: 'mock-integration',
    dimensions: 4,
    async embed(input: string | string[]): Promise<EmbeddingResult> {
      const inputs = Array.isArray(input) ? input : [input];
      const embeddings = inputs.map((text) => {
        // Create embeddings that naturally cluster by topic
        if (text.startsWith('science')) return [0.9, 0.1, 0.0, 0.0];
        if (text.startsWith('sports')) return [0.0, 0.9, 0.1, 0.0];
        if (text.startsWith('music')) return [0.0, 0.0, 0.9, 0.1];
        if (text.startsWith('food')) return [0.1, 0.0, 0.0, 0.9];
        // Default: hash-based
        const hash = text.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
        return [(hash % 10) / 10, ((hash * 3) % 10) / 10, ((hash * 7) % 10) / 10, ((hash * 13) % 10) / 10];
      });
      return { embeddings, model: 'mock', dimensions: 4 };
    },
  };
}

describe('integration: embed → cluster → search', () => {
  it('embeds texts, clusters them, and searches within clusters', async () => {
    const provider = createMockProvider();

    // 20 texts across 4 topics (5 each)
    const texts = [
      ...Array.from({ length: 5 }, (_, i) => `science topic ${i}`),
      ...Array.from({ length: 5 }, (_, i) => `sports topic ${i}`),
      ...Array.from({ length: 5 }, (_, i) => `music topic ${i}`),
      ...Array.from({ length: 5 }, (_, i) => `food topic ${i}`),
    ];

    // Embed all texts
    const result = await provider.embed(texts);
    expect(result.embeddings).toHaveLength(20);

    // Cluster them
    const clusters = clusterEmbeddings(result.embeddings, {
      similarityThreshold: 0.7,
      minClusterSize: 1,
      maxClusters: 6,
    });

    expect(clusters.length).toBeGreaterThanOrEqual(2);

    // Search for a science-like query within the full corpus
    const queryResult = await provider.embed('science query');
    const searchResults = topK(queryResult.embeddings[0], result.embeddings, 3);

    expect(searchResults).toHaveLength(3);
    // Top result should be a science embedding (index 0-4)
    expect(searchResults[0].index).toBeLessThan(5);
  });
});
