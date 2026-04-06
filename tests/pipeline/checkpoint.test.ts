import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createEmbeddingPipeline } from '../../src/pipeline/pipeline';
import type { CheckpointAdapter, CheckpointState } from '../../src/pipeline/checkpoint';
import type { EmbeddingProvider, EmbeddingResult } from '../../src/types';

function createMockProvider(dims: number = 3): EmbeddingProvider & { callLog: string[][] } {
  const provider = {
    name: 'mock',
    dimensions: dims,
    callLog: [] as string[][],
    async embed(input: string | string[]): Promise<EmbeddingResult> {
      const texts = Array.isArray(input) ? input : [input];
      provider.callLog.push([...texts]);
      const embeddings = texts.map((t) => {
        const vec = new Float32Array(dims);
        for (let i = 0; i < dims; i++) vec[i] = (t.length + i) / 10;
        return vec;
      });
      return { embeddings, model: 'mock', dimensions: dims };
    },
  };
  return provider;
}

function createInMemoryCheckpoint(
  initialState: CheckpointState | null = null,
): CheckpointAdapter & { saved: CheckpointState[]; loadCalls: number } {
  const adapter = {
    state: initialState,
    saved: [] as CheckpointState[],
    loadCalls: 0,
    async save(state: CheckpointState) {
      adapter.state = state;
      adapter.saved.push({ ...state, completedIds: [...state.completedIds] });
    },
    async load(): Promise<CheckpointState | null> {
      adapter.loadCalls++;
      return adapter.state;
    },
  };
  return adapter;
}

describe('checkpoint/resume', () => {
  it('checkpoint.save is called at configured interval', async () => {
    const provider = createMockProvider();
    const ckpt = createInMemoryCheckpoint();

    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 2,
      concurrency: 1,
      checkpoint: ckpt,
      checkpointInterval: 2, // save every 2 batches
    });

    const texts = ['a', 'b', 'c', 'd', 'e', 'f'];
    const ids = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6'];
    await pipeline.embed(texts, ids);

    // 6 texts / batchSize 2 = 3 batches
    // Checkpoint at batch 2, then final save after all batches = 2 saves
    // (interval saves at batch 2; final save at end)
    expect(ckpt.saved.length).toBeGreaterThanOrEqual(2);
    // The interval save should have 4 completed IDs (batches 1+2)
    expect(ckpt.saved[0].completedIds).toEqual(['id1', 'id2', 'id3', 'id4']);
  });

  it('checkpoint.load on pipeline start: previously completed IDs are skipped', async () => {
    const provider = createMockProvider();
    const ckpt = createInMemoryCheckpoint({
      completedIds: ['id1', 'id2'],
      totalProcessed: 2,
      timestamp: Date.now(),
    });

    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 2,
      concurrency: 1,
      checkpoint: ckpt,
      checkpointInterval: 100,
    });

    const texts = ['a', 'b', 'c', 'd'];
    const ids = ['id1', 'id2', 'id3', 'id4'];
    await pipeline.embed(texts, ids);

    // First batch (id1, id2) should be skipped
    expect(provider.callLog).toHaveLength(1);
    expect(provider.callLog[0]).toEqual(['c', 'd']);
  });

  it('corrupted checkpoint (load returns null) — pipeline starts fresh', async () => {
    const provider = createMockProvider();
    const ckpt: CheckpointAdapter & { loadCalls: number } = {
      loadCalls: 0,
      async save() {},
      async load() {
        ckpt.loadCalls++;
        return null; // corrupted
      },
    };

    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 2,
      concurrency: 1,
      checkpoint: ckpt,
    });

    const texts = ['a', 'b', 'c', 'd'];
    const ids = ['id1', 'id2', 'id3', 'id4'];
    const result = await pipeline.embed(texts, ids);

    // All batches processed (no skipping)
    expect(provider.callLog).toHaveLength(2);
    expect(result).toHaveLength(4);
    expect(ckpt.loadCalls).toBe(1);
  });

  it('checkpoint state contains completedIds, totalProcessed, timestamp', async () => {
    const provider = createMockProvider();
    const ckpt = createInMemoryCheckpoint();

    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 3,
      concurrency: 1,
      checkpoint: ckpt,
      checkpointInterval: 1,
    });

    const texts = ['a', 'b', 'c'];
    const ids = ['id1', 'id2', 'id3'];
    await pipeline.embed(texts, ids);

    const lastSave = ckpt.saved[ckpt.saved.length - 1];
    expect(lastSave).toHaveProperty('completedIds');
    expect(lastSave).toHaveProperty('totalProcessed');
    expect(lastSave).toHaveProperty('timestamp');
    expect(lastSave.completedIds).toEqual(['id1', 'id2', 'id3']);
    expect(lastSave.totalProcessed).toBe(3);
    expect(typeof lastSave.timestamp).toBe('number');
  });

  it('resume produces same final result as uninterrupted run (idempotent)', async () => {
    const dims = 3;

    // Uninterrupted run
    const provider1 = createMockProvider(dims);
    const pipeline1 = createEmbeddingPipeline(provider1, { batchSize: 2, concurrency: 1 });
    const texts = ['alpha', 'beta', 'gamma', 'delta'];
    const ids = ['id1', 'id2', 'id3', 'id4'];
    const fullResult = await pipeline1.embed(texts, ids);

    // Simulated resume: first 2 items already checkpointed
    const provider2 = createMockProvider(dims);
    const ckpt = createInMemoryCheckpoint({
      completedIds: ['id1', 'id2'],
      totalProcessed: 2,
      timestamp: Date.now(),
    });
    const pipeline2 = createEmbeddingPipeline(provider2, {
      batchSize: 2,
      concurrency: 1,
      checkpoint: ckpt,
    });
    const resumeResult = await pipeline2.embed(texts, ids);

    // Items 3 and 4 should match
    expect(resumeResult[2]).toEqual(fullResult[2]);
    expect(resumeResult[3]).toEqual(fullResult[3]);
    // Only the second batch was called
    expect(provider2.callLog).toHaveLength(1);
  });
});
