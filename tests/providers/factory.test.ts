import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createProvider } from '../../src/providers/factory';
import { ValidationError } from '../../src/types';

// Mock fetch for all provider tests
function mockFetchResponse(data: any, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => data,
  } as Response;
}

describe('createProvider', () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should create OpenAI-compatible provider for type "openai"', () => {
    const provider = createProvider('openai', {
      apiKey: 'key',
      model: 'text-embedding-3-small',
    });
    expect(provider.name).toBe('openai-compatible');
  });

  it('should create Cohere provider for type "cohere"', () => {
    const provider = createProvider('cohere', { apiKey: 'key' });
    expect(provider.name).toBe('cohere');
  });

  it('should create Google Vertex provider for type "google-vertex"', () => {
    const provider = createProvider('google-vertex', {
      projectId: 'proj',
      accessToken: 'tok',
    });
    expect(provider.name).toBe('google-vertex');
  });

  it('should throw ValidationError for unknown type', () => {
    expect(() => createProvider('unknown' as any, {})).toThrow(ValidationError);
    expect(() => createProvider('unknown' as any, {})).toThrow(/unknown provider type/i);
  });

  it('should create Voyage provider as alias for openai-compatible', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('voyage', {
      apiKey: 'key',
      model: 'voyage-3',
    });
    expect(provider.name).toBe('voyageai.com');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.voyageai.com/v1/embeddings');
  });

  it('should create Mistral provider as alias for openai-compatible', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('mistral', {
      apiKey: 'key',
      model: 'mistral-embed',
    });
    expect(provider.name).toBe('mistral.ai');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.mistral.ai/v1/embeddings');
  });

  it('should create Jina provider as alias for openai-compatible', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('jina', {
      apiKey: 'key',
      model: 'jina-embeddings-v3',
    });
    expect(provider.name).toBe('jina.ai');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.jina.ai/v1/embeddings');
  });

  it('should create OpenRouter provider as alias for openai-compatible', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('openrouter', {
      apiKey: 'key',
      model: 'openai/text-embedding-3-small',
    });
    expect(provider.name).toBe('openrouter.ai');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://openrouter.ai/api/v1/embeddings');
  });

  it('should create Together provider with correct base URL', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('together', {
      apiKey: 'key',
      model: 'togethercomputer/m2-bert-80M-8k-retrieval',
    });
    expect(provider.name).toBe('together.xyz');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.together.xyz/v1/embeddings');
  });

  it('should create Fireworks provider with correct base URL', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('fireworks', {
      apiKey: 'key',
      model: 'nomic-ai/nomic-embed-text-v1.5',
    });
    expect(provider.name).toBe('fireworks.ai');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.fireworks.ai/inference/v1/embeddings');
  });

  it('should create Nomic provider with correct base URL', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('nomic', {
      apiKey: 'key',
      model: 'nomic-embed-text-v1.5',
    });
    expect(provider.name).toBe('api-atlas.nomic.ai');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api-atlas.nomic.ai/v1/embeddings');
  });

  it('should create Mixedbread provider with correct base URL', async () => {
    fetchSpy.mockResolvedValue(
      mockFetchResponse({
        data: [{ embedding: [0.1], index: 0 }],
        usage: { total_tokens: 1 },
      }),
    );

    const provider = createProvider('mixedbread', {
      apiKey: 'key',
      model: 'mixedbread-ai/mxbai-embed-large-v1',
    });
    expect(provider.name).toBe('mixedbread.ai');

    await provider.embed('test');
    const [url] = fetchSpy.mock.calls[0];
    expect(url).toBe('https://api.mixedbread.ai/v1/embeddings');
  });
});
