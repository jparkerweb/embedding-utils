import { describe, it, expect } from 'vitest';
import { HNSWIndex } from '../../src/search/hnsw';
import { hdbscan } from '../../src/clustering/hdbscan';
import { recallAtK, ndcg } from '../../src/eval/metrics';
import { topK } from '../../src/search/topk';
import { createEmbeddingStore } from '../../src/store/embedding-store';
import { createEmbeddingPipeline } from '../../src/pipeline/pipeline';
import { chunkByStructure } from '../../src/text/markdown';
import { SearchIndex } from '../../src/search/search-index';
import type { EmbeddingProvider, EmbeddingResult } from '../../src/types';

// ── Mock provider ──────────────────────────────────────────────────────────

/** Deterministic mock provider that groups texts by prefix into distinct vector regions. */
function createMockProvider(dims = 8): EmbeddingProvider {
  return {
    name: 'mock-v03',
    dimensions: dims,
    async embed(input: string | string[]): Promise<EmbeddingResult> {
      const inputs = Array.isArray(input) ? input : [input];
      const embeddings = inputs.map((text) => {
        const vec = new Float32Array(dims);
        // Create separable embeddings based on topic prefixes
        const lower = text.toLowerCase();
        if (lower.includes('machine learning') || lower.includes('neural') || lower.includes('ai')) {
          vec[0] = 0.9; vec[1] = 0.3; vec[2] = 0.1;
        } else if (lower.includes('cooking') || lower.includes('recipe') || lower.includes('pasta')) {
          vec[3] = 0.9; vec[4] = 0.3; vec[5] = 0.1;
        } else if (lower.includes('space') || lower.includes('planet') || lower.includes('nasa')) {
          vec[0] = 0.1; vec[1] = 0.9; vec[6] = 0.3;
        } else {
          // Hash-based fallback
          let h = 0;
          for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
          for (let i = 0; i < dims; i++) {
            vec[i] = Math.abs(((h + i * 997) % 1000) / 1000);
          }
        }
        // Normalize to unit length
        let mag = 0;
        for (let i = 0; i < dims; i++) mag += vec[i] * vec[i];
        mag = Math.sqrt(mag);
        if (mag > 0) for (let i = 0; i < dims; i++) vec[i] /= mag;
        return vec;
      });
      return { embeddings, model: 'mock-v03', dimensions: dims };
    },
  };
}

// ── Task 8.1: embed → HNSW → search → evaluate ────────────────────────────

describe('integration: embed → HNSW → search → evaluate', () => {
  it('produces meaningful recall when compared to brute-force ground truth', async () => {
    const provider = createMockProvider();

    const texts = [
      'machine learning fundamentals',
      'neural network architectures',
      'ai deep learning models',
      'cooking Italian pasta',
      'recipe for homemade bread',
      'cooking Japanese ramen',
      'space exploration by NASA',
      'planet Mars colonization',
      'space shuttle missions',
      'random text about nothing',
    ];

    const { embeddings } = await provider.embed(texts);

    // Build HNSW index
    const hnsw = new HNSWIndex({ metric: 'cosine' });
    for (let i = 0; i < texts.length; i++) {
      hnsw.add(`doc-${i}`, embeddings[i], { text: texts[i] });
    }

    expect(hnsw.size).toBe(texts.length);

    // Query for AI-related content
    const { embeddings: [queryVec] } = await provider.embed('ai and machine learning');

    // HNSW search
    const hnswResults = hnsw.search(queryVec, { topK: 3 });
    const hnswIds = hnswResults.map((r) => r.id);

    // Brute-force ground truth using topK
    const bruteResults = topK(queryVec, embeddings, 3);
    const bruteIds = bruteResults.map((r) => `doc-${r.index}`);

    // HNSW recall vs brute force should be high
    const recall = recallAtK(hnswIds, bruteIds, 3);
    expect(recall).toBeGreaterThanOrEqual(0.66); // at least 2 of 3 match

    // NDCG against known relevance (AI-related docs are 0,1,2)
    const relevance: Record<string, number> = {
      'doc-0': 3,
      'doc-1': 3,
      'doc-2': 3,
    };
    const ndcgScore = ndcg(hnswIds, relevance, 3);
    expect(ndcgScore).toBeGreaterThan(0);

    // All types flow as Float32Array
    for (const result of hnswResults) {
      expect(result.embedding).toBeInstanceOf(Float32Array);
    }
  });
});

// ── Task 8.2: embed → HDBSCAN → inspect ───────────────────────────────────

describe('integration: embed → HDBSCAN → inspect', () => {
  it('discovers clusters with Float32Array centroids and identifies noise', async () => {
    const provider = createMockProvider();

    // 15 texts across 3 topics (5 each)
    const texts = [
      ...Array.from({ length: 5 }, (_, i) => `machine learning topic ${i}`),
      ...Array.from({ length: 5 }, (_, i) => `cooking recipe variation ${i}`),
      ...Array.from({ length: 5 }, (_, i) => `space exploration mission ${i}`),
    ];

    const { embeddings } = await provider.embed(texts);

    const result = hdbscan(embeddings, { minClusterSize: 3, metric: 'euclidean' });

    // Should find at least 2 clusters from the 3 distinct topics
    expect(result.clusters.length).toBeGreaterThanOrEqual(2);

    // Labels array length matches input count
    expect(result.labels).toHaveLength(texts.length);

    // Each label is either a valid cluster index or -1 (noise)
    for (const label of result.labels) {
      expect(label).toBeGreaterThanOrEqual(-1);
      expect(label).toBeLessThan(result.clusters.length);
    }

    // Clusters contain Float32Array centroids and members
    for (const cluster of result.clusters) {
      expect(cluster.centroid).toBeInstanceOf(Float32Array);
      expect(cluster.members.length).toBeGreaterThanOrEqual(3);
      for (const member of cluster.members) {
        expect(member).toBeInstanceOf(Float32Array);
      }
    }

    // Noise members are also Float32Array
    for (const member of result.noise.members) {
      expect(member).toBeInstanceOf(Float32Array);
    }

    // Total points = clusters + noise
    const totalClustered = result.clusters.reduce((sum, c) => sum + c.members.length, 0);
    expect(totalClustered + result.noise.members.length).toBe(texts.length);
  });
});

// ── Task 8.3: pipeline → store → search ────────────────────────────────────

describe('integration: pipeline → store → search', () => {
  it('embeds via pipeline and stores results that are searchable', async () => {
    const provider = createMockProvider();

    // Track progress calls
    const progressCalls: Array<{ completed: number; total: number }> = [];

    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 3,
      onProgress: (info) => {
        progressCalls.push({ completed: info.completed, total: info.total });
      },
    });

    const texts = [
      'machine learning basics',
      'neural network training',
      'cooking pasta recipe',
      'space shuttle launch',
      'ai model deployment',
    ];

    const embeddings = await pipeline.embed(texts);
    expect(embeddings).toHaveLength(texts.length);

    // Progress callback fired
    expect(progressCalls.length).toBeGreaterThan(0);

    // All embeddings are Float32Array
    for (const emb of embeddings) {
      expect(emb).toBeInstanceOf(Float32Array);
    }

    // Store the pipeline output in an EmbeddingStore (using searchByEmbedding)
    const store = createEmbeddingStore({ provider, metric: 'cosine' });

    // Add items using their pre-computed embeddings via the underlying SearchIndex
    // (EmbeddingStore.add calls the provider, so we use a direct approach)
    const index = new SearchIndex({ metric: 'cosine' });
    for (let i = 0; i < texts.length; i++) {
      index.add(`doc-${i}`, embeddings[i], { text: texts[i] });
    }

    // Search
    const { embeddings: [queryVec] } = await provider.embed('ai and machine learning');
    const results = index.search(queryVec, { topK: 2 });

    expect(results.length).toBe(2);
    // Top results should be AI-related
    expect(results[0].score).toBeGreaterThan(0.5);
  });
});

// ── Task 8.4: markdown chunk → embed → search → verify metadata ───────────

describe('integration: markdown chunk → embed → search → verify metadata', () => {
  it('preserves heading metadata end-to-end through chunking, embedding, and search', async () => {
    const provider = createMockProvider();

    const markdown = `# Machine Learning Guide

An overview of machine learning concepts and techniques.

## Neural Networks

Neural networks are the foundation of modern AI and deep learning.

### Training

Training a neural network involves feeding data through layers.

## Cooking Section

This section is about cooking pasta and recipes.

### Italian Pasta

A recipe for making delicious Italian pasta from scratch.
`;

    const chunks = chunkByStructure(markdown, { maxTokens: 200 });

    expect(chunks.length).toBeGreaterThan(0);

    // Embed all chunks
    const chunkTexts = chunks.map((c) => c.text);
    const { embeddings } = await provider.embed(chunkTexts);

    // Build a search index with metadata
    const index = new SearchIndex({ metric: 'cosine' });
    for (let i = 0; i < chunks.length; i++) {
      index.add(`chunk-${i}`, embeddings[i], {
        headings: chunks[i].metadata.headings,
        type: chunks[i].metadata.type,
        offset: chunks[i].metadata.offset,
      });
    }

    // Search for AI-related content
    const { embeddings: [queryVec] } = await provider.embed('neural network ai learning');
    const results = index.search(queryVec, { topK: 3 });

    expect(results.length).toBeGreaterThan(0);

    // Verify metadata is preserved on results
    for (const result of results) {
      expect(result.metadata).toBeDefined();
      expect(result.metadata!.headings).toBeDefined();
      expect(Array.isArray(result.metadata!.headings)).toBe(true);
      expect(result.metadata!.type).toBeDefined();
      expect(typeof result.metadata!.offset).toBe('number');
    }

    // Verify embeddings are Float32Array
    for (const result of results) {
      expect(result.embedding).toBeInstanceOf(Float32Array);
    }

    // Filtered search: only chunks under "Cooking Section"
    const cookingResults = index.search(queryVec, {
      topK: 10,
      filter: (item) => {
        const headings = item.metadata?.headings as string[] | undefined;
        return headings?.includes('Cooking Section') ?? false;
      },
    });

    // All cooking results should have Cooking Section in headings
    for (const result of cookingResults) {
      const headings = result.metadata!.headings as string[];
      expect(headings).toContain('Cooking Section');
    }
  });
});
