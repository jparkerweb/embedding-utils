import type {
  EmbeddingProvider,
  EmbeddingResult,
  EmbedOptions,
  GoogleVertexConfig,
} from '../types';
import { ProviderError } from '../types';
import { retryWithBackoff, autoBatch } from './shared';

const DEFAULT_LOCATION = 'us-central1';
const DEFAULT_MODEL = 'text-embedding-005';
const MAX_BATCH_SIZE = 5;

/**
 * Creates a Google Vertex AI embedding provider.
 * @param config - Configuration: projectId, accessToken (string or async getter), optional location, model, retry
 * @returns An EmbeddingProvider that calls the Vertex AI predict API
 * @example
 * const provider = createGoogleVertexProvider({
 *   projectId: 'my-project', accessToken: 'ya29...'
 * });
 * const result = await provider.embed('hello world');
 */
export function createGoogleVertexProvider(
  config: GoogleVertexConfig,
): EmbeddingProvider {
  const location = config.location ?? DEFAULT_LOCATION;
  const model = config.model ?? DEFAULT_MODEL;
  const retryConfig = config.retry ?? { maxRetries: 3, baseDelay: 1000, maxDelay: 30000 };

  const endpoint = `https://${location}-aiplatform.googleapis.com/v1/projects/${config.projectId}/locations/${location}/publishers/google/models/${model}:predict`;

  async function resolveToken(): Promise<string> {
    if (typeof config.accessToken === 'function') {
      return config.accessToken();
    }
    return config.accessToken;
  }

  async function embedBatch(
    batch: string[],
    signal?: AbortSignal,
  ): Promise<{ embeddings: number[][] }> {
    const token = await resolveToken();

    const result = await retryWithBackoff(
      async () => {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            instances: batch.map((text) => ({ content: text })),
          }),
          signal,
        });

        if (!response.ok) {
          throw new ProviderError(
            `HTTP ${response.status}: ${response.statusText}`,
            'google-vertex',
            response.status,
          );
        }

        return response.json();
      },
      retryConfig,
      signal,
    );

    const embeddings = result.predictions.map(
      (p: any) => p.embeddings.values,
    );
    return { embeddings };
  }

  return {
    name: 'google-vertex',
    dimensions: config.retry ? null : null,

    async embed(
      input: string | string[],
      options?: EmbedOptions,
    ): Promise<EmbeddingResult> {
      const inputs = Array.isArray(input) ? input : [input];
      let allEmbeddings: number[][];

      if (inputs.length <= MAX_BATCH_SIZE) {
        const result = await embedBatch(inputs, options?.signal);
        allEmbeddings = result.embeddings;
      } else {
        allEmbeddings = await autoBatch(inputs, MAX_BATCH_SIZE, async (batch) => {
          const result = await embedBatch(batch, options?.signal);
          return result.embeddings;
        });
      }

      return {
        embeddings: allEmbeddings,
        model,
        dimensions: allEmbeddings[0]?.length ?? 0,
      };
    },
  };
}
