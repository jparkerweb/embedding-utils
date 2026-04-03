import { validateEmbeddings } from '../internal/validation';
import type { SerializationMetadata } from '../types';

type SerializeFormat = 'json' | 'binary' | 'base64';

/** Result returned by deserialize, including optional v2 metadata. */
export interface DeserializeResult {
  embeddings: number[][];
  metadata?: SerializationMetadata;
}

// ─── Cross-platform base64 helpers (no Buffer dependency) ───────────────────

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return globalThis.btoa(binary);
}

function base64ToUint8(base64: string): Uint8Array {
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

// ─── V1 binary format (legacy, read-only) ──────────────────────────────────
// Layout: [count:u32LE] then per embedding: [dims:u32LE][float32LE * dims]

function fromBinaryV1(data: Uint8Array): number[][] {
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

// ─── V2 binary format ───────────────────────────────────────────────────────
// Layout:
//   [version: u8 = 2]
//   [flags: u8]           bit 0 = has metadata
//   [count: u32LE]
//   [dims: u32LE]         single dims header (all embeddings same dimension)
//   [metadata_length: u32LE]  (only if flag bit 0 set)
//   [metadata_json: utf8]     (only if flag bit 0 set)
//   [embeddings: float32LE * count * dims]

const V2_VERSION = 2;
const FLAG_HAS_METADATA = 0x01;

function toBinaryV2(embeddings: number[][], metadata?: SerializationMetadata): Uint8Array {
  const count = embeddings.length;
  const dims = embeddings[0].length;

  let metadataBytes: Uint8Array | undefined;
  let flags = 0;

  if (metadata) {
    flags |= FLAG_HAS_METADATA;
    const encoder = new TextEncoder();
    metadataBytes = encoder.encode(JSON.stringify(metadata));
  }

  // Header: version(1) + flags(1) + count(4) + dims(4) = 10
  let totalBytes = 10;
  if (metadataBytes) {
    totalBytes += 4 + metadataBytes.length; // metadata_length + metadata_json
  }
  totalBytes += count * dims * 4; // embedding data

  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  let offset = 0;

  view.setUint8(offset, V2_VERSION);
  offset += 1;
  view.setUint8(offset, flags);
  offset += 1;
  view.setUint32(offset, count, true);
  offset += 4;
  view.setUint32(offset, dims, true);
  offset += 4;

  if (metadataBytes) {
    view.setUint32(offset, metadataBytes.length, true);
    offset += 4;
    new Uint8Array(buffer, offset, metadataBytes.length).set(metadataBytes);
    offset += metadataBytes.length;
  }

  for (const emb of embeddings) {
    for (let i = 0; i < dims; i++) {
      view.setFloat32(offset, emb[i], true);
      offset += 4;
    }
  }

  return new Uint8Array(buffer);
}

function fromBinaryV2(data: Uint8Array): DeserializeResult {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;

  // skip version byte (already checked by caller)
  offset += 1;
  const flags = view.getUint8(offset);
  offset += 1;
  const count = view.getUint32(offset, true);
  offset += 4;
  const dims = view.getUint32(offset, true);
  offset += 4;

  let metadata: SerializationMetadata | undefined;
  if (flags & FLAG_HAS_METADATA) {
    const metadataLength = view.getUint32(offset, true);
    offset += 4;
    const decoder = new TextDecoder();
    metadata = JSON.parse(
      decoder.decode(new Uint8Array(data.buffer, data.byteOffset + offset, metadataLength)),
    );
    offset += metadataLength;
  }

  const embeddings: number[][] = [];
  for (let i = 0; i < count; i++) {
    const emb = new Array<number>(dims);
    for (let j = 0; j < dims; j++) {
      emb[j] = view.getFloat32(offset, true);
      offset += 4;
    }
    embeddings.push(emb);
  }

  return { embeddings, metadata };
}

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * Serializes embeddings to JSON string, binary Uint8Array, or base64 string.
 *
 * Uses binary format v2 by default (version byte, single dims header, optional metadata).
 *
 * @param embeddings - Array of embedding vectors to serialize
 * @param format - Output format: 'json', 'binary', or 'base64'
 * @param metadata - Optional metadata to include in binary/base64 formats (v2 only)
 * @returns Serialized data (string for json/base64, Uint8Array for binary)
 * @throws {ValidationError} If the embeddings array is empty
 * @example
 * serialize([[1, 2], [3, 4]], 'json'); // '[[1,2],[3,4]]'
 * serialize([[1, 2]], 'base64'); // base64 string
 * serialize([[1, 2]], 'binary', { model: 'text-embedding-3-small' });
 */
export function serialize(
  embeddings: number[][],
  format: SerializeFormat,
  metadata?: SerializationMetadata,
): string | Uint8Array {
  validateEmbeddings(embeddings);

  switch (format) {
    case 'json':
      return JSON.stringify(embeddings);

    case 'binary':
      return toBinaryV2(embeddings, metadata);

    case 'base64': {
      const binary = toBinaryV2(embeddings, metadata);
      return uint8ToBase64(binary);
    }
  }
}

/**
 * Deserializes embeddings from a serialized format.
 *
 * Auto-detects binary format version: if the first byte is `2`, parses as v2;
 * otherwise falls back to v1 legacy format.
 *
 * @param data - Serialized data (string for json/base64, Uint8Array for binary)
 * @param format - Input format: 'json', 'binary', or 'base64'
 * @returns Object with embeddings array and optional metadata (v2 only)
 * @example
 * deserialize('[[1,2],[3,4]]', 'json'); // { embeddings: [[1, 2], [3, 4]] }
 * deserialize(uint8Data, 'binary'); // { embeddings: [...], metadata: { model: '...' } }
 */
export function deserialize(
  data: string | Uint8Array,
  format: SerializeFormat,
): DeserializeResult {
  switch (format) {
    case 'json':
      return { embeddings: JSON.parse(data as string) };

    case 'binary': {
      const bytes = data as Uint8Array;
      if (bytes.length > 0 && bytes[0] === V2_VERSION) {
        return fromBinaryV2(bytes);
      }
      return { embeddings: fromBinaryV1(bytes) };
    }

    case 'base64': {
      const bytes = base64ToUint8(data as string);
      if (bytes.length > 0 && bytes[0] === V2_VERSION) {
        return fromBinaryV2(bytes);
      }
      return { embeddings: fromBinaryV1(bytes) };
    }
  }
}
