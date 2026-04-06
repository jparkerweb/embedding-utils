import { describe, it, expect } from 'vitest';
import { HNSWIndex } from '../../src/search/hnsw';
import { SearchIndex } from '../../src/search/search-index';

function randomVector(dims: number): Float32Array {
  const v = new Float32Array(dims);
  for (let i = 0; i < dims; i++) v[i] = Math.random() * 2 - 1;
  return v;
}

describe('HNSWIndex — performance benchmark', () => {
  const DIMS = 128;
  const COUNT = 10_000;
  const NUM_QUERIES = 100;

  // This test is intentionally slow — skip in CI, enable manually
  it.skip('10k vectors: >95% recall@10 and >10x speedup vs brute-force', () => {
    const vectors = Array.from({ length: COUNT }, () => randomVector(DIMS));
    const queries = Array.from({ length: NUM_QUERIES }, () => randomVector(DIMS));

    // Build HNSW index
    const hnsw = new HNSWIndex({ metric: 'cosine', M: 16, efConstruction: 200 });
    for (let i = 0; i < COUNT; i++) {
      hnsw.add(`v${i}`, vectors[i]);
    }

    // Build brute-force index
    const bf = new SearchIndex({ metric: 'cosine' });
    for (let i = 0; i < COUNT; i++) {
      bf.add(`v${i}`, vectors[i]);
    }

    // Measure HNSW query time
    const hnswStart = performance.now();
    const hnswResults: string[][] = [];
    for (const q of queries) {
      const results = hnsw.search(q, { topK: 10, efSearch: 50 });
      hnswResults.push(results.map((r) => r.id));
    }
    const hnswTime = performance.now() - hnswStart;

    // Measure brute-force query time
    const bfStart = performance.now();
    const bfResults: string[][] = [];
    for (const q of queries) {
      const results = bf.search(q, { topK: 10 });
      bfResults.push(results.map((r) => r.id));
    }
    const bfTime = performance.now() - bfStart;

    // Calculate recall
    let totalRecall = 0;
    for (let i = 0; i < NUM_QUERIES; i++) {
      const hnswIds = new Set(hnswResults[i]);
      const bfIds = bfResults[i];
      let overlap = 0;
      for (const id of bfIds) {
        if (hnswIds.has(id)) overlap++;
      }
      totalRecall += overlap / 10;
    }
    const avgRecall = totalRecall / NUM_QUERIES;

    const speedup = bfTime / hnswTime;

    console.log(`HNSW Benchmark (${COUNT} vectors, ${DIMS} dims, ${NUM_QUERIES} queries):`);
    console.log(`  Recall@10: ${(avgRecall * 100).toFixed(1)}%`);
    console.log(`  HNSW avg query: ${(hnswTime / NUM_QUERIES).toFixed(2)}ms`);
    console.log(`  Brute-force avg query: ${(bfTime / NUM_QUERIES).toFixed(2)}ms`);
    console.log(`  Speedup: ${speedup.toFixed(1)}x`);

    expect(avgRecall).toBeGreaterThan(0.95);
    expect(speedup).toBeGreaterThan(10);
  });
});
