/* tslint:disable */
/* eslint-disable */

/* auto-generated by NAPI-RS */

export interface RwkvInvocation {
  nThreads: number
  nTokPredict: number
  topK: number
  topP: number
  temp: number
  repeatPenalty: number
  stopSequence?: string
  prompt: string
}
export interface RwkvContextParams {
  nCtx: number
  nParts: number
  seed: number
  f16Kv: boolean
  logitsAll: boolean
  vocabOnly: boolean
  useMlock: boolean
  embedding: boolean
  useMmap: boolean
}
export const enum TokenizeResultType {
  Error = 'Error',
  Data = 'Data'
}
export interface TokenizeResult {
  type: TokenizeResultType
  data: Array<number>
}
export interface InferenceToken {
  token: string
  completed: boolean
}
export const enum InferenceResultType {
  Error = 'Error',
  Data = 'Data',
  End = 'End'
}
export interface InferenceResult {
  type: InferenceResultType
  data?: InferenceToken
  message?: string
}
export const enum EmbeddingResultType {
  Error = 'Error',
  Data = 'Data'
}
export interface EmbeddingResult {
  type: EmbeddingResultType
  data: Array<number>
}
export class LLama {
  static load(modelPath: string, tokenizerPath: string, params: number, enableLogger: boolean): LLama
  getWordEmbedding(input: RwkvInvocation, callback: (result: EmbeddingResult) => void): void
  tokenize(params: string, callback: (result: TokenizeResult) => void): void
  inference(input: RwkvInvocation, callback: (result: InferenceResult) => void): void
}