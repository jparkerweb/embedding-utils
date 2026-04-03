import type {
  EmbeddingProvider,
  EmbeddingResult,
  EmbedOptions,
  OpenAICompatibleConfig,
} from '../types';
import { ProviderError, ValidationError } from '../types';
import { retryWithBackoff, autoBatch, createTimeoutSignal, wrapTimeoutError } from './shared';
import type { OpenAIEmbeddingResponse } from './types';

const DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const DEFAULT_MAX_BATCH_SIZE = 2048;

function extractProviderName(baseUrl: string): string {
  try {
    const hostname = new URL(baseUrl).hostname;
    // Strip leading 'api.' prefix
    const name = hostname.replace(/^api\./, '');
    return name;
  } catch {
    return 'openai-compatible';
  }
}

/**
 * Creates an OpenAI-compatible embedding provider (works with OpenAI, Voyage, Mistral, Jina, OpenRouter).
 * @param config - Configuration: apiKey, model, optional baseUrl, dimensions, maxBatchSize, retry
 * @returns An EmbeddingProvider that calls the OpenAI-compatible API
 * @example
 * const provider = createOpenAICompatibleProvider({
 *   apiKey: 'sk-...', model: 'text-embedding-3-small'
 * });
 * const result = await provider.embed(['hello', 'world']);
 */
export function createOpenAICompatibleProvider(
  config: OpenAICompatibleConfig,
): EmbeddingProvider {
  const baseUrl = config.baseUrl ?? DEFAULT_BASE_URL;
  const maxBatchSize = config.maxBatchSize ?? DEFAULT_MAX_BATCH_SIZE;
  const retryConfig = config.retry ?? { maxRetries: 3, baseDelay: 1000, maxDelay: 30000 };
  const providerName =
    baseUrl === DEFAULT_BASE_URL ? 'openai-compatible' : extractProviderName(baseUrl);

  async function embedBatch(
    batch: string[],
    signal?: AbortSignal,
  ): Promise<{ embeddings: number[][]; tokens: number }> {
    const fetchSignal = createTimeoutSignal(config.timeout, signal);
    const result = await retryWithBackoff(
      async () => {
        const body: Record<string, unknown> = {
          model: config.model,
          input: batch,
        };
        if (config.dimensions != null) {
          body.dimensions = config.dimensions;
        }

        const response = await fetch(`${baseUrl}/embeddings`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${config.apiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
          signal: fetchSignal,
        });

        if (!response.ok) {
          let body: string;
          try {
            body = await response.text();
          } catch {
            body = response.statusText;
          }
          throw new ProviderError(
            `HTTP ${response.status}: ${body}`,
            providerName,
            response.status,
          );
        }

        try {
          return await response.json() as OpenAIEmbeddingResponse;
        } catch {
          throw new ProviderError(
            'Failed to parse API response',
            providerName,
            response.status,
          );
        }
      },
      retryConfig,
      signal,
    );

    if (!Array.isArray(result.data) || !result.data[0]?.embedding) {
      throw new ProviderError(
        'Unexpected API response structure from openai-compatible',
        providerName,
      );
    }

    const embeddings = result.data.map((d) => d.embedding);
    const tokens = result.usage?.total_tokens ?? 0;
    return { embeddings, tokens };
  }

  return {
    name: providerName,
    dimensions: config.dimensions ?? null,

    async embed(
      input: string | string[],
      options?: EmbedOptions,
    ): Promise<EmbeddingResult> {
      const inputs = Array.isArray(input) ? input : [input];
      for (const text of inputs) {
        if (text.trim().length === 0) {
          throw new ValidationError('Cannot embed empty string');
        }
      }

      try {
        let allEmbeddings: number[][];
        let totalTokens = 0;

        if (inputs.length <= maxBatchSize) {
          const result = await embedBatch(inputs, options?.signal);
          allEmbeddings = result.embeddings;
          totalTokens = result.tokens;
        } else {
          allEmbeddings = await autoBatch(inputs, maxBatchSize, async (batch) => {
            const result = await embedBatch(batch, options?.signal);
            totalTokens += result.tokens;
            return result.embeddings;
          });
        }

        return {
          embeddings: allEmbeddings,
          model: config.model,
          dimensions: allEmbeddings[0]?.length ?? 0,
          usage: { tokens: totalTokens },
        };
      } catch (error) {
        wrapTimeoutError(error, providerName, config.timeout);
      }
    },
  };
}
