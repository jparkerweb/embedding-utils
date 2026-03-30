import { describe, it, expect, vi } from 'vitest';
import { retryWithBackoff, autoBatch } from '../../src/providers/shared';
import { ProviderError } from '../../src/types';

describe('retryWithBackoff', () => {
  it('should return result on first success', async () => {
    const fn = vi.fn().mockResolvedValue('ok');
    const result = await retryWithBackoff(fn, { maxRetries: 3 });
    expect(result).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should retry on 429 status', async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new ProviderError('rate limited', 'test', 429))
      .mockResolvedValue('ok');

    const result = await retryWithBackoff(fn, {
      maxRetries: 3,
      baseDelay: 1,
      maxDelay: 10,
    });
    expect(result).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it('should retry on 5xx status', async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new ProviderError('server error', 'test', 500))
      .mockRejectedValueOnce(new ProviderError('server error', 'test', 503))
      .mockResolvedValue('ok');

    const result = await retryWithBackoff(fn, {
      maxRetries: 3,
      baseDelay: 1,
      maxDelay: 10,
    });
    expect(result).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should NOT retry on 4xx (except 429)', async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new ProviderError('bad request', 'test', 400));

    await expect(
      retryWithBackoff(fn, { maxRetries: 3, baseDelay: 1 }),
    ).rejects.toThrow('bad request');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should NOT retry on non-ProviderError', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('unknown'));

    await expect(
      retryWithBackoff(fn, { maxRetries: 3, baseDelay: 1 }),
    ).rejects.toThrow('unknown');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should respect maxRetries', async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new ProviderError('server error', 'test', 500));

    await expect(
      retryWithBackoff(fn, { maxRetries: 2, baseDelay: 1, maxDelay: 10 }),
    ).rejects.toThrow('server error');
    // Initial call + 2 retries = 3 total
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should increase delays exponentially', async () => {
    const delays: number[] = [];
    const originalSetTimeout = globalThis.setTimeout;
    vi.spyOn(globalThis, 'setTimeout').mockImplementation((fn: any, delay?: number) => {
      if (delay !== undefined && delay > 0) delays.push(delay);
      // Execute immediately
      fn();
      return 0 as any;
    });

    const fnMock = vi
      .fn()
      .mockRejectedValueOnce(new ProviderError('error', 'test', 500))
      .mockRejectedValueOnce(new ProviderError('error', 'test', 500))
      .mockResolvedValue('ok');

    await retryWithBackoff(fnMock, {
      maxRetries: 3,
      baseDelay: 100,
      maxDelay: 10000,
    });

    expect(delays.length).toBe(2);
    // Second delay should be larger than first (exponential + jitter)
    // baseDelay * 2^0 + jitter for first, baseDelay * 2^1 + jitter for second
    // With jitter, first is ~100-130, second is ~200-260
    expect(delays[1]).toBeGreaterThan(delays[0]);

    vi.restoreAllMocks();
  });

  it('should stop retries when AbortSignal is aborted', async () => {
    const controller = new AbortController();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new ProviderError('error', 'test', 500))
      .mockImplementation(async () => {
        return 'ok';
      });

    // Abort before retry happens
    vi.spyOn(globalThis, 'setTimeout').mockImplementation((cb: any, _delay?: number) => {
      controller.abort();
      cb();
      return 0 as any;
    });

    await expect(
      retryWithBackoff(fn, { maxRetries: 3, baseDelay: 1 }, controller.signal),
    ).rejects.toThrow(/abort/i);

    vi.restoreAllMocks();
  });
});

describe('autoBatch', () => {
  it('should split input into batches of batchSize', async () => {
    const batches: string[][] = [];
    const fn = vi.fn().mockImplementation(async (batch: string[]) => {
      batches.push(batch);
      return batch.map(() => [1, 2, 3]);
    });

    await autoBatch(['a', 'b', 'c', 'd', 'e'], 2, fn);
    expect(batches).toEqual([['a', 'b'], ['c', 'd'], ['e']]);
  });

  it('should process batches sequentially', async () => {
    const order: number[] = [];
    const fn = vi.fn().mockImplementation(async (batch: string[]) => {
      const idx = order.length;
      order.push(idx);
      // Small delay to verify sequential execution
      await new Promise((r) => setTimeout(r, 1));
      return batch.map(() => [1]);
    });

    await autoBatch(['a', 'b', 'c'], 1, fn);
    expect(order).toEqual([0, 1, 2]);
  });

  it('should concatenate results from all batches', async () => {
    const fn = vi.fn().mockImplementation(async (batch: string[]) => {
      return batch.map((_, i) => [i + 1]);
    });

    const result = await autoBatch(['a', 'b', 'c', 'd'], 2, fn);
    // First batch: [0+1], [1+1] = [[1],[2]], second batch: [0+1],[1+1] = [[1],[2]]
    expect(result).toEqual([[1], [2], [1], [2]]);
    expect(result.length).toBe(4);
  });

  it('should handle single batch (input <= batchSize)', async () => {
    const fn = vi.fn().mockImplementation(async (batch: string[]) => {
      return batch.map(() => [1, 2]);
    });

    const result = await autoBatch(['a', 'b'], 10, fn);
    expect(result).toEqual([[1, 2], [1, 2]]);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should handle empty input', async () => {
    const fn = vi.fn();
    const result = await autoBatch([], 5, fn);
    expect(result).toEqual([]);
    expect(fn).not.toHaveBeenCalled();
  });
});
