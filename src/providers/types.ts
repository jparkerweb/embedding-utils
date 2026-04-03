/**
 * Typed response interfaces for cloud provider APIs.
 *
 * These types replace `any` at the JSON parse boundary in each provider.
 * Only fields actually accessed by provider code are typed.
 * @internal
 */

/** Response shape from the OpenAI-compatible `/v1/embeddings` endpoint. */
export interface OpenAIEmbeddingResponse {
  data: Array<{ embedding: number[]; index: number }>;
  usage?: { total_tokens: number };
  model?: string;
}

/** Response shape from the Cohere v2 `/embed` endpoint. */
export interface CohereEmbeddingResponse {
  embeddings: {
    float: number[][];
  };
  meta?: {
    billed_units?: {
      input_tokens?: number;
    };
  };
}

/** Response shape from the Google Vertex AI `:predict` endpoint. */
export interface GoogleVertexEmbeddingResponse {
  predictions: Array<{
    embeddings: {
      values: number[];
    };
  }>;
}
