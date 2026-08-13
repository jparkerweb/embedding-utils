import { describe, it, expect, vi } from 'vitest';
import { createInstanceRegistry } from '../../src/internal/instance-registry';

describe('createInstanceRegistry', () => {
  it('constructs once per key and shares the promise', async () => {
    const registry = createInstanceRegistry<string>(4);
    const factory = vi.fn(async () => 'instance');

    const a = registry.getOrCreate('k', factory);
    const b = registry.getOrCreate('k', factory);

    expect(a).toBe(b);
    expect(await a).toBe('instance');
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it('constructs separately for distinct keys', async () => {
    const registry = createInstanceRegistry<number>(4);
    let n = 0;
    const factory = vi.fn(async () => ++n);

    expect(await registry.getOrCreate('a', factory)).toBe(1);
    expect(await registry.getOrCreate('b', factory)).toBe(2);
    expect(factory).toHaveBeenCalledTimes(2);
    expect(registry.size).toBe(2);
  });

  it('dedupes concurrent construction of the same key', async () => {
    const registry = createInstanceRegistry<string>(4);
    let resolveIt!: (v: string) => void;
    const factory = vi.fn(() => new Promise<string>((resolve) => (resolveIt = resolve)));

    const a = registry.getOrCreate('k', factory);
    const b = registry.getOrCreate('k', factory); // while still pending
    resolveIt('done');

    expect(await a).toBe('done');
    expect(await b).toBe('done');
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it('evicts a failed construction so the next call retries', async () => {
    const registry = createInstanceRegistry<string>(4);
    const factory = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce('recovered');

    await expect(registry.getOrCreate('k', factory)).rejects.toThrow('boom');
    expect(registry.size).toBe(0);
    expect(await registry.getOrCreate('k', factory)).toBe('recovered');
    expect(factory).toHaveBeenCalledTimes(2);
  });

  it('evicts the least-recently-used entry beyond the bound', async () => {
    const registry = createInstanceRegistry<string>(2);
    const factory = vi.fn(async () => 'v');

    await registry.getOrCreate('a', factory);
    await registry.getOrCreate('b', factory);
    // Touch 'a' so 'b' is now least-recently-used.
    await registry.getOrCreate('a', factory);
    await registry.getOrCreate('c', factory); // evicts 'b'

    expect(registry.size).toBe(2);
    expect(factory).toHaveBeenCalledTimes(3);
    // 'a' survived the eviction (it was touched more recently than 'b').
    await registry.getOrCreate('a', factory);
    expect(factory).toHaveBeenCalledTimes(3);
    // 'b' was evicted: re-requesting reconstructs (and evicts 'c').
    await registry.getOrCreate('b', factory);
    expect(factory).toHaveBeenCalledTimes(4);
  });

  it('clear() empties the registry and returns settled instances', async () => {
    const registry = createInstanceRegistry<string>(4);
    await registry.getOrCreate('a', async () => 'one');
    await registry.getOrCreate('b', async () => 'two');

    const cleared = await registry.clear();

    expect(cleared.sort()).toEqual(['one', 'two']);
    expect(registry.size).toBe(0);
  });

  it('clear() skips rejected constructions instead of throwing', async () => {
    const registry = createInstanceRegistry<string>(4);
    await registry.getOrCreate('good', async () => 'ok');
    // A pending-then-failing construction present at clear() time.
    const failing = registry.getOrCreate('bad', () => Promise.reject(new Error('nope')));

    const cleared = await registry.clear();

    expect(cleared).toEqual(['ok']);
    await expect(failing).rejects.toThrow('nope');
  });
});
