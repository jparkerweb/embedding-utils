import type {
  EmbeddingProvider,
  EmbeddingResult,
  EmbedOptions,
  CohereConfig,
} from '../types';
import { ProviderError, ValidationError } from '../types';
import { toFloat32 } from '../internal/vector-utils';
import { retryWithBackoff, autoBatch, createTimeoutSignal, wrapTimeoutError } from './shared';
import type { CohereEmbeddingResponse } from './types';

const COHERE_API_URL = 'https://api.cohere.com/v2/embed';
const DEFAULT_MODEL = 'embed-v4.0';
const MAX_BATCH_SIZE = 96;

function mapInputType(inputType?: 'document' | 'query'): string {
  if (inputType === 'query') return 'search_query';
  return 'search_document';
}

/**
 * Creates a Cohere embedding provider.
 * @param config - Configuration: apiKey, optional model (default 'embed-v4.0'), dimensions, retry
 * @returns An EmbeddingProvider that calls the Cohere embed API
 * @example
 * const provider = createCohereProvider({ apiKey: 'co-...' });
 * const result = await provider.embed('hello world', { inputType: 'query' });
 */
export function createCohereProvider(config: CohereConfig): EmbeddingProvider {
  const model = config.model ?? DEFAULT_MODEL;
  const retryConfig = config.retry ?? { maxRetries: 3, baseDelay: 1000, maxDelay: 30000 };

  async function embedBatch(
    batch: string[],
    inputType: string,
    signal?: AbortSignal,
  ): Promise<{ embeddings: number[][]; tokens: number }> {
    const fetchSignal = createTimeoutSignal(config.timeout, signal);
    const result = await retryWithBackoff(
      async () => {
        const response = await fetch(COHERE_API_URL, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${config.apiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model,
            texts: batch,
            input_type: inputType,
            embedding_types: ['float'],
          }),
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
            'cohere',
            response.status,
          );
        }

        try {
          return await response.json() as CohereEmbeddingResponse;
        } catch {
          throw new ProviderError(
            'Failed to parse API response',
            'cohere',
            response.status,
          );
        }
      },
      retryConfig,
      signal,
    );

    if (!result.embeddings?.float || !Array.isArray(result.embeddings.float)) {
      throw new ProviderError(
        'Unexpected API response structure from cohere',
        'cohere',
      );
    }

    const embeddings = result.embeddings.float;
    const tokens = result.meta?.billed_units?.input_tokens ?? 0;
    return { embeddings, tokens };
  }

  return {
    name: 'cohere',
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
      const inputType = mapInputType(options?.inputType);

      try {
        let allEmbeddings: number[][];
        let totalTokens = 0;

        if (inputs.length <= MAX_BATCH_SIZE) {
          const result = await embedBatch(inputs, inputType, options?.signal);
          allEmbeddings = result.embeddings;
          totalTokens = result.tokens;
        } else {
          allEmbeddings = await autoBatch(inputs, MAX_BATCH_SIZE, async (batch) => {
            const result = await embedBatch(batch, inputType, options?.signal);
            totalTokens += result.tokens;
            return result.embeddings;
          });
        }

        return {
          embeddings: allEmbeddings.map(toFloat32),
          model,
          dimensions: allEmbeddings[0]?.length ?? 0,
          usage: { tokens: totalTokens },
        };
      } catch (error) {
        wrapTimeoutError(error, 'cohere', config.timeout);
      }
    },
  };
}
