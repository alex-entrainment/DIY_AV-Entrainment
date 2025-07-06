/* tslint:disable */
/* eslint-disable */
export function start_stream(track_json_str: string, sample_rate: number, start_time: number): void;
export function update_track(track_json_str: string): void;
export function enable_gpu(enable: boolean): void;
export function process_block(frame_count: number): Float32Array;
export function current_step(): number;
export function elapsed_samples(): bigint;
export function pause_stream(): void;
export function resume_stream(): void;
export function start_from(position: number): void;
export function stop_stream(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly start_stream: (a: number, b: number, c: number, d: number) => void;
  readonly update_track: (a: number, b: number) => void;
  readonly enable_gpu: (a: number) => void;
  readonly process_block: (a: number) => any;
  readonly current_step: () => number;
  readonly elapsed_samples: () => bigint;
  readonly stop_stream: () => void;
  readonly start_from: (a: number) => void;
  readonly pause_stream: () => void;
  readonly resume_stream: () => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
