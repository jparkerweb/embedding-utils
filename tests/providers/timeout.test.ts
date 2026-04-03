import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createOpenAICompatibleProvider } from '../../src/providers/openai-compatible';
import { createCohereProvider } from '../../src/providers/cohere';
import { createGoogleVertexProvider } from '../../src/providers/google';
import { ProviderError } from '../../src/types';

describe('provider timeouts', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // Create a fetch that never resolves (hangs until aborted)
  function stubHangingFetch() {
    const hangingFetch = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(init.signal!.reason);
          });
        }),
    );
    vi.stubGlobal('fetch', hangingFetch);
  }

  it('openai-compatible provider throws ProviderError on timeout', async () => {
    stubHangingFetch();

    const provider = createOpenAICompatibleProvider({
      apiKey: 'test-key',
      model: 'test-model',
      timeout: 50,
      retry: { maxRetries: 0 },
    });

    try {
      await provider.embed('hello');
      expect.unreachable('should have thrown');
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ProviderError);
      expect((err as ProviderError).message).toMatch(/timed out/i);
      expect((err as ProviderError).message).toContain('50ms');
    }
  }, 10000);

  it('cohere provider throws ProviderError on timeout', async () => {
    stubHangingFetch();

    const provider = createCohereProvider({
      apiKey: 'test-key',
      timeout: 50,
      retry: { maxRetries: 0 },
    });

    try {
      await provider.embed('hello');
      expect.unreachable('should have thrown');
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ProviderError);
      expect((err as ProviderError).message).toMatch(/timed out/i);
    }
  }, 10000);

  it('google-vertex provider throws ProviderError on timeout', async () => {
    stubHangingFetch();

    const provider = createGoogleVertexProvider({
      projectId: 'test-project',
      accessToken: 'test-token',
      timeout: 50,
      retry: { maxRetries: 0 },
    });

    try {
      await provider.embed('hello');
      expect.unreachable('should have thrown');
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ProviderError);
      expect((err as ProviderError).message).toMatch(/timed out/i);
    }
  }, 10000);
});
