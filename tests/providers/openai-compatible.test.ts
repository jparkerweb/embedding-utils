import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createOpenAICompatibleProvider } from '../../src/providers/openai-compatible';
import { ProviderError } from '../../src/types';

function mockFetchResponse(data: any, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => data,
  } as Response;
}

describe('createOpenAICompatibleProvider', () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const baseConfig = {
    apiKey: 'test-key',
    model: 'text-embedding-3-small',
  };

  const successResponse = {
    data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
    usage: { total_tokens: 5 },
  };

  it('should have correct provider name and dimensions', () => {
    const provider = createOpenAICompatibleProvider(baseConfig);
    expect(provider.name).toBe('openai-compatible');
    expect(provider.dimensions).toBeNull();
  });

  it('should have dimensions from config', () => {
    const provider = createOpenAICompatibleProvider({ ...baseConfig, dimensions: 256 });
    expect(provider.dimensions).toBe(256);
  });

  it('should send correct request shape for single string input', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider(baseConfig);
    await provider.embed('hello world');

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, options] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.openai.com/v1/embeddings');
    expect(options.method).toBe('POST');

    const body = JSON.parse(options.body);
    expect(body.model).toBe('text-embedding-3-small');
    expect(body.input).toEqual(['hello world']);
  });

  it('should send correct Authorization header', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider(baseConfig);
    await provider.embed('test');

    const [, options] = fetchSpy.mock.calls[0];
    expect(options.headers['Authorization']).toBe('Bearer test-key');
    expect(options.headers['Content-Type']).toBe('application/json');
  });

  it('should handle batch string[] input', async () => {
    const batchResponse = {
      data: [
        { embedding: [0.1, 0.2], index: 0 },
        { embedding: [0.3, 0.4], index: 1 },
      ],
      usage: { total_tokens: 10 },
    };
    fetchSpy.mockResolvedValue(mockFetchResponse(batchResponse));

    const provider = createOpenAICompatibleProvider(baseConfig);
    const result = await provider.embed(['hello', 'world']);

    expect(result.embeddings).toEqual([[0.1, 0.2], [0.3, 0.4]]);
    expect(result.usage).toEqual({ tokens: 10 });
  });

  it('should parse response correctly', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider(baseConfig);
    const result = await provider.embed('test');

    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
    expect(result.model).toBe('text-embedding-3-small');
    expect(result.dimensions).toBe(3);
    expect(result.usage).toEqual({ tokens: 5 });
  });

  it('should use custom baseUrl for Voyage', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      baseUrl: 'https://api.voyageai.com/v1',
    });
    await provider.embed('test');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.voyageai.com/v1/embeddings');
  });

  it('should use custom baseUrl for Mistral', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      baseUrl: 'https://api.mistral.ai/v1',
    });
    await provider.embed('test');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.mistral.ai/v1/embeddings');
  });

  it('should use custom baseUrl for Jina', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      baseUrl: 'https://api.jina.ai/v1',
    });
    await provider.embed('test');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.jina.ai/v1/embeddings');
  });

  it('should use custom baseUrl for OpenRouter', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      baseUrl: 'https://openrouter.ai/api/v1',
    });
    await provider.embed('test');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://openrouter.ai/api/v1/embeddings');
  });

  it('should pass dimensions option in request body', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({ ...baseConfig, dimensions: 128 });
    await provider.embed('test');

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.dimensions).toBe(128);
  });

  it('should not include dimensions when not configured', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider(baseConfig);
    await provider.embed('test');

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.dimensions).toBeUndefined();
  });

  it('should retry on 429 then succeed', async () => {
    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse({ error: 'rate limited' }, 429))
      .mockResolvedValueOnce(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      retry: { maxRetries: 2, baseDelay: 1, maxDelay: 10 },
    });
    const result = await provider.embed('test');

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
  });

  it('should retry on 500 then succeed', async () => {
    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse({ error: 'server error' }, 500))
      .mockResolvedValueOnce(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      retry: { maxRetries: 2, baseDelay: 1, maxDelay: 10 },
    });
    const result = await provider.embed('test');

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
  });

  it('should NOT retry on 400', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'bad request' }, 400));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      retry: { maxRetries: 2, baseDelay: 1 },
    });

    await expect(provider.embed('test')).rejects.toThrow(ProviderError);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('should forward AbortSignal to fetch', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const controller = new AbortController();
    const provider = createOpenAICompatibleProvider(baseConfig);
    await provider.embed('test', { signal: controller.signal });

    const [, options] = fetchSpy.mock.calls[0];
    // Signal is combined with timeout signal, so it's a composite signal
    // Verify that aborting the user signal also aborts the fetch signal
    expect(options.signal.aborted).toBe(false);
    controller.abort();
    expect(options.signal.aborted).toBe(true);
  });

  it('should throw ProviderError on non-retryable HTTP errors', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'forbidden' }, 403));

    const provider = createOpenAICompatibleProvider(baseConfig);

    try {
      await provider.embed('test');
      expect.unreachable('should have thrown');
    } catch (err: any) {
      expect(err).toBeInstanceOf(ProviderError);
      expect(err.statusCode).toBe(403);
      expect(err.provider).toBe('openai-compatible');
    }
  });

  it('should auto-batch when input exceeds maxBatchSize', async () => {
    const makeResponse = (count: number) => ({
      data: Array.from({ length: count }, (_, i) => ({ embedding: [i], index: i })),
      usage: { total_tokens: count },
    });

    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse(makeResponse(2)))
      .mockResolvedValueOnce(mockFetchResponse(makeResponse(1)));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      maxBatchSize: 2,
      retry: { maxRetries: 0 },
    });
    const result = await provider.embed(['a', 'b', 'c']);

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings.length).toBe(3);
    expect(result.usage).toEqual({ tokens: 3 });
  });

  it('should extract provider name from custom baseUrl hostname', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createOpenAICompatibleProvider({
      ...baseConfig,
      baseUrl: 'https://api.voyageai.com/v1',
    });

    expect(provider.name).toBe('voyageai.com');
  });
});
