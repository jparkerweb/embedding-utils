import { describe, it, expect } from 'vitest';
import { recallAtK, ndcg, mrr, meanAveragePrecision } from '../../src/eval/metrics';

// ── Task 6.1: recallAtK ──────────────────────────────────────────────────────

describe('recallAtK', () => {
  it('computes recall for partially matching results', () => {
    // retrieved ['a','b','c'], relevant ['a','c','d'], k=3 → 2/3
    expect(recallAtK(['a', 'b', 'c'], ['a', 'c', 'd'], 3)).toBeCloseTo(2 / 3);
  });

  it('k=1, first result relevant → recall = 1.0', () => {
    expect(recallAtK(['a', 'b'], new Set(['a']), 1)).toBe(1.0);
  });

  it('k=1, first result irrelevant → recall = 0.0', () => {
    expect(recallAtK(['b', 'a'], new Set(['a']), 1)).toBe(0.0);
  });

  it('empty retrieved → 0.0', () => {
    expect(recallAtK([], ['a', 'b'])).toBe(0.0);
  });

  it('empty relevant → 0.0', () => {
    expect(recallAtK(['a', 'b'], [])).toBe(0.0);
  });

  it('perfect recall (all relevant in retrieved)', () => {
    expect(recallAtK(['a', 'b', 'c'], ['a', 'b', 'c'])).toBe(1.0);
  });
});

// ── Task 6.2: ndcg ───────────────────────────────────────────────────────────

describe('ndcg', () => {
  it('perfect order → NDCG = 1.0', () => {
    // Items a,b,c,d with relevance 3,2,1,0 in perfect order
    const retrieved = ['a', 'b', 'c', 'd'];
    const scores = { a: 3, b: 2, c: 1, d: 0 };
    expect(ndcg(retrieved, scores)).toBeCloseTo(1.0);
  });

  it('reverse order → NDCG < 1.0', () => {
    const retrieved = ['d', 'c', 'b', 'a'];
    const scores = { a: 3, b: 2, c: 1, d: 0 };
    // Compute DCG for reverse: (2^0-1)/log2(2) + (2^1-1)/log2(3) + (2^2-1)/log2(4) + (2^3-1)/log2(5)
    // = 0/1 + 1/1.585 + 3/2 + 7/2.322 = 0 + 0.6309 + 1.5 + 3.0145 = 5.1454
    // IDCG (perfect): (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) + (2^0-1)/log2(5)
    // = 7/1 + 3/1.585 + 1/2 + 0/2.322 = 7 + 1.8928 + 0.5 + 0 = 9.3928
    const dcg = 0 / Math.log2(2) + (Math.pow(2, 1) - 1) / Math.log2(3) + (Math.pow(2, 2) - 1) / Math.log2(4) + (Math.pow(2, 3) - 1) / Math.log2(5);
    const idcg = (Math.pow(2, 3) - 1) / Math.log2(2) + (Math.pow(2, 2) - 1) / Math.log2(3) + (Math.pow(2, 1) - 1) / Math.log2(4) + 0;
    const expected = dcg / idcg;
    const result = ndcg(retrieved, scores);
    expect(result).toBeCloseTo(expected);
    expect(result).toBeLessThan(1.0);
  });

  it('k cutoff limits evaluation to first k results', () => {
    const retrieved = ['a', 'b', 'c', 'd'];
    const scores = { a: 3, b: 2, c: 1, d: 0 };
    // k=2: only first 2 results evaluated
    const result = ndcg(retrieved, scores, 2);
    expect(result).toBeCloseTo(1.0); // first 2 are best 2 in order
  });

  it('all irrelevant → NDCG = 0.0', () => {
    const retrieved = ['a', 'b', 'c'];
    const scores = { a: 0, b: 0, c: 0 };
    expect(ndcg(retrieved, scores)).toBe(0);
  });

  it('single result, relevant → NDCG = 1.0', () => {
    expect(ndcg(['a'], { a: 3 })).toBeCloseTo(1.0);
  });
});

// ── Task 6.3: mrr ────────────────────────────────────────────────────────────

describe('mrr', () => {
  it('first result relevant → MRR = 1.0', () => {
    expect(mrr(['a', 'b', 'c'], new Set(['a']))).toBe(1.0);
  });

  it('relevant at position 3 → MRR = 1/3', () => {
    expect(mrr(['x', 'y', 'a'], ['a'])).toBeCloseTo(1 / 3);
  });

  it('no relevant found → MRR = 0.0', () => {
    expect(mrr(['x', 'y', 'z'], ['a'])).toBe(0.0);
  });

  it('multiple relevant — uses first occurrence only', () => {
    // 'b' at position 2 is first relevant
    expect(mrr(['x', 'b', 'a'], ['a', 'b'])).toBeCloseTo(1 / 2);
  });
});

// ── Task 6.4: meanAveragePrecision ───────────────────────────────────────────

describe('meanAveragePrecision', () => {
  it('computes MAP for known precision-recall curve', () => {
    // retrieved: [r, n, r, n, r] where r=relevant, n=not
    // relevant = {a, c, e}
    // position 1 (a): precision = 1/1
    // position 3 (c): precision = 2/3
    // position 5 (e): precision = 3/5
    // MAP = (1 + 2/3 + 3/5) / 3
    const retrieved = ['a', 'x', 'c', 'y', 'e'];
    const relevant = ['a', 'c', 'e'];
    const expected = (1 + 2 / 3 + 3 / 5) / 3;
    expect(meanAveragePrecision(retrieved, relevant)).toBeCloseTo(expected);
  });

  it('perfect ranking → MAP = 1.0', () => {
    expect(meanAveragePrecision(['a', 'b', 'c'], ['a', 'b', 'c'])).toBeCloseTo(1.0);
  });

  it('no relevant documents → MAP = 0.0', () => {
    expect(meanAveragePrecision(['a', 'b'], [])).toBe(0.0);
  });

  it('single query with all results relevant → MAP = 1.0', () => {
    expect(meanAveragePrecision(['a', 'b'], ['a', 'b'])).toBeCloseTo(1.0);
  });
});
