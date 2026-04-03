import { describe, it, expect } from 'vitest';
import { promisePool } from '../../src/internal/concurrency';

describe('promisePool', () => {
  it('returns results in original order', async () => {
    const tasks = [
      () => Promise.resolve('a'),
      () => Promise.resolve('b'),
      () => Promise.resolve('c'),
    ];
    const results = await promisePool(tasks, 2);
    expect(results).toEqual(['a', 'b', 'c']);
  });

  it('handles empty task array', async () => {
    const results = await promisePool([], 3);
    expect(results).toEqual([]);
  });

  it('respects max concurrency', async () => {
    let active = 0;
    let maxActive = 0;
    const maxConcurrency = 2;

    const tasks = Array.from({ length: 6 }, (_, i) => async () => {
      active++;
      maxActive = Math.max(maxActive, active);
      await new Promise((r) => setTimeout(r, 10));
      active--;
      return i;
    });

    const results = await promisePool(tasks, maxConcurrency);
    expect(results).toEqual([0, 1, 2, 3, 4, 5]);
    expect(maxActive).toBeLessThanOrEqual(maxConcurrency);
  });

  it('propagates the first error and stops starting new tasks', async () => {
    const executed: number[] = [];

    const tasks = [
      async () => {
        executed.push(0);
        return 'ok';
      },
      async () => {
        executed.push(1);
        throw new Error('fail');
      },
      async () => {
        // Should not execute after error
        await new Promise((r) => setTimeout(r, 50));
        executed.push(2);
        return 'ok';
      },
    ];

    await expect(promisePool(tasks, 1)).rejects.toThrow('fail');
    expect(executed).toEqual([0, 1]);
  });

  it('works with concurrency of 1 (sequential)', async () => {
    const order: number[] = [];
    const tasks = [0, 1, 2].map((i) => async () => {
      order.push(i);
      return i;
    });
    const results = await promisePool(tasks, 1);
    expect(results).toEqual([0, 1, 2]);
    expect(order).toEqual([0, 1, 2]);
  });

  it('handles concurrency greater than task count', async () => {
    const tasks = [() => Promise.resolve(1), () => Promise.resolve(2)];
    const results = await promisePool(tasks, 10);
    expect(results).toEqual([1, 2]);
  });
});
