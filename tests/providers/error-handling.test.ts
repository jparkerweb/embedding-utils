import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createOpenAICompatibleProvider } from '../../src/providers/openai-compatible';
import { createCohereProvider } from '../../src/providers/cohere';
import { createGoogleVertexProvider } from '../../src/providers/google';
import { ProviderError, ValidationError } from '../../src/types';

function mockFetchResponse(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => data,
    text: async () => JSON.stringify(data),
  } as Response;
}

function mockFetchNonJson(body: string, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => {
      throw new SyntaxError('Unexpected token');
    },
    text: async () => body,
  } as Response;
}

describe('Provider error handling', () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('OpenAI-compatible', () => {
    const config = { apiKey: 'key', model: 'test-model', retry: { maxRetries: 0 } };

    it('throws ProviderError for non-2xx response', async () => {
      fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'forbidden' }, 403));
      const provider = createOpenAICompatibleProvider(config);
      await expect(provider.embed('test')).rejects.toThrow(ProviderError);
    });

    it('throws ProviderError when response is not valid JSON', async () => {
      fetchSpy.mockResolvedValue(mockFetchNonJson('<html>Gateway Timeout</html>'));
      const provider = createOpenAICompatibleProvider(config);
      await expect(provider.embed('test')).rejects.toThrow('Failed to parse API response');
    });

    it('throws ProviderError for malformed response structure', async () => {
      fetchSpy.mockResolvedValue(mockFetchResponse({ unexpected: true }));
      const provider = createOpenAICompatibleProvider(config);
      await expect(provider.embed('test')).rejects.toThrow('Unexpected API response structure');
    });

    it('throws ProviderError with correct provider field', async () => {
      fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'bad' }, 400));
      const provider = createOpenAICompatibleProvider(config);
      try {
        await provider.embed('test');
        expect.unreachable('should throw');
      } catch (err) {
        expect(err).toBeInstanceOf(ProviderError);
        expect((err as ProviderError).provider).toBe('openai-compatible');
        expect((err as ProviderError).statusCode).toBe(400);
      }
    });

    it('throws ValidationError for empty string input', async () => {
      const provider = createOpenAICompatibleProvider(config);
      await expect(provider.embed('')).rejects.toThrow(ValidationError);
      await expect(provider.embed('  ')).rejects.toThrow('Cannot embed empty string');
    });

    it('throws ValidationError for empty string in batch', async () => {
      const provider = createOpenAICompatibleProvider(config);
      await expect(provider.embed(['hello', '', 'world'])).rejects.toThrow(ValidationError);
    });
  });

  describe('Cohere', () => {
    const config = { apiKey: 'key', retry: { maxRetries: 0 } };

    it('throws ProviderError for non-2xx response', async () => {
      fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'unauthorized' }, 401));
      const provider = createCohereProvider(config);
      await expect(provider.embed('test')).rejects.toThrow(ProviderError);
    });

    it('throws ProviderError when response is not valid JSON', async () => {
      fetchSpy.mockResolvedValue(mockFetchNonJson('not json'));
      const provider = createCohereProvider(config);
      await expect(provider.embed('test')).rejects.toThrow('Failed to parse API response');
    });

    it('throws ProviderError for malformed response structure', async () => {
      fetchSpy.mockResolvedValue(mockFetchResponse({ embeddings: {} }));
      const provider = createCohereProvider(config);
      await expect(provider.embed('test')).rejects.toThrow('Unexpected API response structure');
    });

    it('throws ValidationError for empty string input', async () => {
      const provider = createCohereProvider(config);
      await expect(provider.embed('')).rejects.toThrow(ValidationError);
    });
  });

  describe('Google Vertex', () => {
    const config = { projectId: 'proj', accessToken: 'tok', retry: { maxRetries: 0 } };

    it('throws ProviderError for non-2xx response', async () => {
      fetchSpy.mockResolvedValue(mockFetchResponse({ error: 'not found' }, 404));
      const provider = createGoogleVertexProvider(config);
      await expect(provider.embed('test')).rejects.toThrow(ProviderError);
    });

    it('throws ProviderError when response is not valid JSON', async () => {
      fetchSpy.mockResolvedValue(mockFetchNonJson('bad response'));
      const provider = createGoogleVertexProvider(config);
      await expect(provider.embed('test')).rejects.toThrow('Failed to parse API response');
    });

    it('throws ProviderError for malformed response structure', async () => {
      fetchSpy.mockResolvedValue(mockFetchResponse({ predictions: [] }));
      const provider = createGoogleVertexProvider(config);
      await expect(provider.embed('test')).rejects.toThrow('Unexpected API response structure');
    });

    it('throws ValidationError for empty string input', async () => {
      const provider = createGoogleVertexProvider(config);
      await expect(provider.embed('')).rejects.toThrow(ValidationError);
    });
  });
});
