import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createGoogleVertexProvider } from '../../src/providers/google';
import { ProviderError } from '../../src/types';

function mockFetchResponse(data: any, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => data,
  } as Response;
}

describe('createGoogleVertexProvider', () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const baseConfig = {
    projectId: 'my-project',
    accessToken: 'test-token',
  };

  const successResponse = {
    predictions: [
      { embeddings: { values: [0.1, 0.2, 0.3] } },
    ],
  };

  it('should have correct provider name and dimensions', () => {
    const provider = createGoogleVertexProvider(baseConfig);
    expect(provider.name).toBe('google-vertex');
    expect(provider.dimensions).toBeNull();
  });

  it('should send correct URL format with defaults', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider(baseConfig);
    await provider.embed('hello');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe(
      'https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/text-embedding-005:predict',
    );
  });

  it('should use custom location', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider({
      ...baseConfig,
      location: 'europe-west1',
    });
    await provider.embed('hello');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toContain('europe-west1-aiplatform.googleapis.com');
    expect(url).toContain('locations/europe-west1');
  });

  it('should use custom model', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider({
      ...baseConfig,
      model: 'text-embedding-004',
    });
    await provider.embed('hello');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toContain('models/text-embedding-004:predict');
  });

  it('should send correct Authorization header with string token', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider(baseConfig);
    await provider.embed('test');

    const [, options] = fetchSpy.mock.calls[0];
    expect(options.headers['Authorization']).toBe('Bearer test-token');
    expect(options.headers['Content-Type']).toBe('application/json');
  });

  it('should resolve token from async function each request', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    let callCount = 0;
    const tokenFn = vi.fn(async () => {
      callCount++;
      return `dynamic-token-${callCount}`;
    });

    const provider = createGoogleVertexProvider({
      ...baseConfig,
      accessToken: tokenFn,
    });

    await provider.embed('first');
    expect(fetchSpy.mock.calls[0][1].headers['Authorization']).toBe(
      'Bearer dynamic-token-1',
    );

    await provider.embed('second');
    expect(fetchSpy.mock.calls[1][1].headers['Authorization']).toBe(
      'Bearer dynamic-token-2',
    );

    expect(tokenFn).toHaveBeenCalledTimes(2);
  });

  it('should send correct request body with instances', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider(baseConfig);
    await provider.embed('hello');

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.instances).toEqual([{ content: 'hello' }]);
  });

  it('should handle batch input', async () => {
    const batchResponse = {
      predictions: [
        { embeddings: { values: [0.1, 0.2] } },
        { embeddings: { values: [0.3, 0.4] } },
      ],
    };
    fetchSpy.mockResolvedValue(mockFetchResponse(batchResponse));

    const provider = createGoogleVertexProvider(baseConfig);
    const result = await provider.embed(['a', 'b']);

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body);
    expect(body.instances).toEqual([{ content: 'a' }, { content: 'b' }]);

    expect(result.embeddings).toEqual([[0.1, 0.2], [0.3, 0.4]]);
  });

  it('should parse predictions response correctly', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider(baseConfig);
    const result = await provider.embed('test');

    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
    expect(result.model).toBe('text-embedding-005');
    expect(result.dimensions).toBe(3);
  });

  it('should use default location us-central1', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider(baseConfig);
    await provider.embed('test');

    const [url] = fetchSpy.mock.calls[0];
    expect(url).toContain('us-central1');
  });

  it('should use default model text-embedding-005', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider(baseConfig);
    const result = await provider.embed('test');

    expect(result.model).toBe('text-embedding-005');
  });

  it('should retry on 429 then succeed', async () => {
    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse({ error: 'rate limited' }, 429))
      .mockResolvedValueOnce(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider({
      ...baseConfig,
      retry: { maxRetries: 2, baseDelay: 1, maxDelay: 10 },
    });
    const result = await provider.embed('test');

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
  });

  it('should retry on 5xx then succeed', async () => {
    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse({ error: 'server error' }, 503))
      .mockResolvedValueOnce(mockFetchResponse(successResponse));

    const provider = createGoogleVertexProvider({
      ...baseConfig,
      retry: { maxRetries: 2, baseDelay: 1, maxDelay: 10 },
    });
    const result = await provider.embed('test');

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
  });

  it('should throw ProviderError on non-retryable errors', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'forbidden' }, 403));

    const provider = createGoogleVertexProvider(baseConfig);

    try {
      await provider.embed('test');
      expect.unreachable('should have thrown');
    } catch (err: any) {
      expect(err).toBeInstanceOf(ProviderError);
      expect(err.status).toBe(403);
      expect(err.provider).toBe('google-vertex');
    }
  });

  it('should forward AbortSignal to fetch', async () => {
    fetchSpy.mockResolvedValue(mockFetchResponse(successResponse));

    const controller = new AbortController();
    const provider = createGoogleVertexProvider(baseConfig);
    await provider.embed('test', { signal: controller.signal });

    const [, options] = fetchSpy.mock.calls[0];
    expect(options.signal).toBe(controller.signal);
  });

  it('should auto-batch when input exceeds 5', async () => {
    const makeResponse = (count: number) => ({
      predictions: Array.from({ length: count }, () => ({
        embeddings: { values: [0.1] },
      })),
    });

    const inputs = Array.from({ length: 8 }, (_, i) => `text-${i}`);

    fetchSpy
      .mockResolvedValueOnce(mockFetchResponse(makeResponse(5)))
      .mockResolvedValueOnce(mockFetchResponse(makeResponse(3)));

    const provider = createGoogleVertexProvider({
      ...baseConfig,
      retry: { maxRetries: 0 },
    });
    const result = await provider.embed(inputs);

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(result.embeddings.length).toBe(8);
  });
});
