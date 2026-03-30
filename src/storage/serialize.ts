import { ValidationError } from '../types';

type SerializeFormat = 'json' | 'binary' | 'base64';

function validateEmbeddings(embeddings: number[][]): void {
  if (embeddings.length === 0) {
    throw new ValidationError('Embeddings array must be non-empty');
  }
}

function toBinary(embeddings: number[][]): Uint8Array {
  // Format: 4 bytes (uint32) embedding count, then per embedding:
  //   4 bytes (uint32) dimension count + dims*4 bytes float32 values
  let totalBytes = 4; // embedding count header
  for (const emb of embeddings) {
    totalBytes += 4 + emb.length * 4; // dim header + float32 values
  }

  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  let offset = 0;

  view.setUint32(offset, embeddings.length, true);
  offset += 4;

  for (const emb of embeddings) {
    view.setUint32(offset, emb.length, true);
    offset += 4;
    for (let i = 0; i < emb.length; i++) {
      view.setFloat32(offset, emb[i], true);
      offset += 4;
    }
  }

  return new Uint8Array(buffer);
}

function fromBinary(data: Uint8Array): number[][] {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;

  const count = view.getUint32(offset, true);
  offset += 4;

  const embeddings: number[][] = [];
  for (let i = 0; i < count; i++) {
    const dims = view.getUint32(offset, true);
    offset += 4;
    const emb = new Array<number>(dims);
    for (let j = 0; j < dims; j++) {
      emb[j] = view.getFloat32(offset, true);
      offset += 4;
    }
    embeddings.push(emb);
  }

  return embeddings;
}

/**
 * Serializes embeddings to JSON string, binary Uint8Array, or base64 string.
 * @param embeddings - Array of embedding vectors to serialize
 * @param format - Output format: 'json', 'binary', or 'base64'
 * @returns Serialized data (string for json/base64, Uint8Array for binary)
 * @throws {ValidationError} If the embeddings array is empty
 * @example
 * serialize([[1, 2], [3, 4]], 'json'); // '[[1,2],[3,4]]'
 * serialize([[1, 2]], 'base64'); // base64 string
 */
export function serialize(
  embeddings: number[][],
  format: SerializeFormat,
): string | Uint8Array {
  validateEmbeddings(embeddings);

  switch (format) {
    case 'json':
      return JSON.stringify(embeddings);

    case 'binary':
      return toBinary(embeddings);

    case 'base64': {
      const binary = toBinary(embeddings);
      return Buffer.from(binary).toString('base64');
    }
  }
}

/**
 * Deserializes embeddings from a serialized format.
 * @param data - Serialized data (string for json/base64, Uint8Array for binary)
 * @param format - Input format: 'json', 'binary', or 'base64'
 * @returns Array of embedding vectors
 * @example
 * deserialize('[[1,2],[3,4]]', 'json'); // [[1, 2], [3, 4]]
 */
export function deserialize(
  data: string | Uint8Array,
  format: SerializeFormat,
): number[][] {
  switch (format) {
    case 'json':
      return JSON.parse(data as string);

    case 'binary':
      return fromBinary(data as Uint8Array);

    case 'base64': {
      const buffer = Buffer.from(data as string, 'base64');
      return fromBinary(new Uint8Array(buffer));
    }
  }
}
