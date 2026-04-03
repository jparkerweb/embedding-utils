import { describe, it, expect } from 'vitest';
import {
  cosineSimilarity,
  topK,
  aboveThreshold,
  normalize,
  ValidationError,
} from '../src';
import { validateVector } from '../src/internal/validation';

describe('edge cases', () => {
  describe('NaN in input vector', () => {
    it('validateVector throws ValidationError for NaN input', () => {
      expect(() => validateVector([1, NaN, 3])).toThrow(ValidationError);
    });

    it('validateVector throws ValidationError for NaN at any position', () => {
      expect(() => validateVector([NaN, 2, 3])).toThrow(ValidationError);
    });
  });

  describe('Infinity in input vector', () => {
    it('validateVector throws ValidationError for Infinity input', () => {
      expect(() => validateVector([1, Infinity, 3])).toThrow(ValidationError);
    });

    it('validateVector throws ValidationError for -Infinity input', () => {
      expect(() => validateVector([1, -Infinity, 3])).toThrow(ValidationError);
    });
  });

  describe('empty corpus in topK', () => {
    it('returns empty array for empty corpus', () => {
      const result = topK([1, 0, 0], [], 5);
      expect(result).toEqual([]);
    });
  });

  describe('empty corpus in aboveThreshold', () => {
    it('returns empty array for empty corpus', () => {
      const result = aboveThreshold([1, 0, 0], [], 0.5);
      expect(result).toEqual([]);
    });
  });

  describe('zero-length vector in similarity', () => {
    it('cosineSimilarity throws for empty vectors', () => {
      expect(() => cosineSimilarity([], [])).toThrow(ValidationError);
    });
  });

  describe('concurrent provider calls', () => {
    it('two independent topK calls resolve independently', async () => {
      const corpus = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ];
      const query1 = [1, 0, 0];
      const query2 = [0, 1, 0];

      const [result1, result2] = await Promise.all([
        Promise.resolve(topK(query1, corpus, 1)),
        Promise.resolve(topK(query2, corpus, 1)),
      ]);

      expect(result1[0].index).toBe(0);
      expect(result2[0].index).toBe(1);
    });
  });

  describe('very large k in topK', () => {
    it('k > corpus.length returns all items sorted by score', () => {
      const corpus = [
        [1, 0, 0],
        [0.9, 0.1, 0],
        [0, 1, 0],
      ];
      const query = [1, 0, 0];

      const results = topK(query, corpus, 100);
      expect(results).toHaveLength(corpus.length);
      // Should be sorted descending by score
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
      }
    });

    it('k equal to corpus.length returns all items', () => {
      const corpus = [[1, 0], [0, 1]];
      const results = topK([1, 0], corpus, 2);
      expect(results).toHaveLength(2);
    });
  });

  describe('normalize edge cases', () => {
    it('normalize throws for empty vector', () => {
      expect(() => normalize([])).toThrow(ValidationError);
    });
  });
});
