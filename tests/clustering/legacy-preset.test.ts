import { describe, it, expect } from 'vitest';
import { clusterEmbeddings, getPreset } from '../../src';

describe('legacy clustering preset', () => {
  it('produces a single cluster containing all input embeddings', () => {
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0.5, 0.5, 0],
      [0, 0.5, 0.5],
    ];

    const clusters = clusterEmbeddings(embeddings, getPreset('legacy'));

    expect(clusters).toHaveLength(1);
    expect(clusters[0].members).toHaveLength(embeddings.length);
    expect(clusters[0].size).toBe(embeddings.length);
  });

  it('works with different embedding dimensions', () => {
    const embeddings = [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
    ];

    const clusters = clusterEmbeddings(embeddings, getPreset('legacy'));

    expect(clusters).toHaveLength(1);
    expect(clusters[0].size).toBe(2);
  });
});
