import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createCohereProvider } from '../../src/providers/cohere';
import { ProviderError } from '../../src/types';

function mockFetchResponse(data: any, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => data,
  } as Response;
}

describe('createCohereProvider', () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const baseConfig = {
    apiKey: 'cohere-test-key',
  };

  const successResponse = {
    embeddings: {
      float: [[0.1, 0.2, 0.3]],
    },
    meta: { billed_units: { input_tokens: 5 } },
  };

  it('should have correct provider name and dimensions', () => {
    const provider = createCohereProvider(baseConfig);
    expect(provider.name).toBe('cohere');
    expect(provider.dimensions).toBeNull();
  });

  it('should have dimensions from config', () => {
    const provider = createCohereProvider({ ...baseConfig, dimensions: 512 });
    expect(provider.dimensions).toBe(512);
  });

  it('should send correct request shape', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createCohereProvider(baseConfig);
    await provider.embed('hello');

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, options] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.cohere.com/v2/embed');
    expect(options.method).toBe('POST');

    const body = JSON.parse(options.body);
    expect(body.model).toBe('embed-v4.0');
    expect(body.texts).toEqual(['hello']);
    expect(body.input_type).toBe('search_document');
    expect(body.embedding_types).toEqual(['float']);
  });

  it('should send correct Authorization header', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createCohereProvider(baseConfig);
    await provider.embed('test');

    const [, options] = fetchSpy.mock.calls[0];
    expect(options.headers['Authorization']).toBe('Bearer cohere-test-key');
    expect(options.headers['Content-Type']).toBe('application/json');
  });

  it('should map inputType document to search_document', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createCohereProvider(baseConfig);
    await provider.embed('test', { inputType: 'document' });

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.input_type).toBe('search_document');
  });

  it('should map inputType query to search_query', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createCohereProvider(baseConfig);
    await provider.embed('test', { inputType: 'query' });

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.input_type).toBe('search_query');
  });

  it('should default inputType to search_document', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createCohereProvider(baseConfig);
    await provider.embed('test');

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.input_type).toBe('search_document');
  });

  it('should parse embeddings.float response', async () => {
    const response = {
      embeddings: {
        float: [[0.1, 0.2], [0.3, 0.4]],
      },
      meta: { billed_units: { input_tokens: 8 } },
    };
    fetchSpy.mockResolvedValue(mockFetchResponse(response));

    const provider = createCohereProvider(baseConfig);
    const result = await provider.embed(['a', 'b']);

    expect(result.embeddings).toEqual([[0.1, 0.2], [0.3, 0.4]]);
    expect(result.usage).toEqual({ tokens: 8 });
  });

  it('should use default model embed-v4.0', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createCohereProvider(baseConfig);
    const result = await provider.embed('test');

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.model).toBe('embed-v4.0');
    expect(result.model).toBe('embed-v4.0');
  });

  it('should use custom model', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createCohereProvider({ ...baseConfig, model: 'embed-english-v3.0' });
    const result = await provider.embed('test');

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.model).toBe('embed-english-v3.0');
    expect(result.model).toBe('embed-english-v3.0');
  });

  it('should retry on 429 then succeed', async () => {
    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse({ error: 'rate limited' }, 429))
      .mockResolvedValueOnce(mockFetchResponse(successResponse));

    const provider = createCohereProvider({
      ...baseConfig,
      retry: { maxRetries: 2, baseDelay: 1, maxDelay: 10 },
    });
    const result = await provider.embed('test');

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
  });

  it('should retry on 5xx then succeed', async () => {
    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse({ error: 'server error' }, 502))
      .mockResolvedValueOnce(mockFetchResponse(successResponse));

    const provider = createCohereProvider({
      ...baseConfig,
      retry: { maxRetries: 2, baseDelay: 1, maxDelay: 10 },
    });
    const result = await provider.embed('test');

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
  });

  it('should throw ProviderError on non-retryable errors', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'bad' }, 400));

    const provider = createCohereProvider(baseConfig);

    try {
      await provider.embed('test');
      expect.unreachable('should have thrown');
    } catch (err: any) {
      expect(err).toBeInstanceOf(ProviderError);
      expect(err.statusCode).toBe(400);
      expect(err.provider).toBe('cohere');
    }
  });

  it('should forward AbortSignal to fetch', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const controller = new AbortController();
    const provider = createCohereProvider(baseConfig);
    await provider.embed('test', { signal: controller.signal });

    const [, options] = fetchSpy.mock.calls[0];
    // Signal is combined with timeout signal, so it's a composite signal
    expect(options.signal.aborted).toBe(false);
    controller.abort();
    expect(options.signal.aborted).toBe(true);
  });

  it('should auto-batch when input exceeds 96', async () => {
    const makeResponse = (count: number) => ({
      embeddings: { float: Array.from({ length: count }, () => [0.1]) },
      meta: { billed_units: { input_tokens: count } },
    });

    // Create 100 inputs to exceed the 96 batch limit
    const inputs = Array.from({ length: 100 }, (_, i) => `text-${i}`);

    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse(makeResponse(96)))
      .mockResolvedValueOnce(mockFetchResponse(makeResponse(4)));

    const provider = createCohereProvider({
      ...baseConfig,
      retry: { maxRetries: 0 },
    });
    const result = await provider.embed(inputs);

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings.length).toBe(100);
    expect(result.usage).toEqual({ tokens: 100 });
  });
});
