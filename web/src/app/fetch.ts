// apiClient.ts
/* eslint-disable @typescript-eslint/no-explicit-any */
// For Vercel deployment, use relative URL. For local dev, use localhost:5000
const API_BASE_URL =
  typeof window !== "undefined" && window.location.hostname === "localhost"
    ? "http://127.0.0.1:5000"
    : "";
    
type Position = "rb" | "qb" | "wr" | "te";

export interface FetchDataPayload {
  position: Position | string;
  type?: string;   // e.g. "predictions"
  model?: string;  // e.g. "xgb"
  limit?: number;  // pagination limit
  offset?: number; // pagination offset
  time?: string;
  [key: string]: unknown; // <-- this allows any string key
}

export interface FetchFeaturesPayload {
  position: Position | string;
  time?: string;
  [key: string]: unknown; // <-- this allows any string key
}

/**
 * Generic response type for JSON responses.
 * Use a more specific type for your prediction records if known.
 */
export type JsonArray = Array<Record<string, unknown>>;
export type FeaturesResponse = string[];

/**
 * Paginated response type for fetch_data endpoint
 */
export interface PaginatedResponse {
  data: JsonArray;
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

/**
 * Throws an Error on non-2xx responses. Attempts to include server body text in the message.
 */
async function postJSON<T = any>(
  endpoint: string,
  payload: Record<string, unknown> = {},
  opts?: { signal?: AbortSignal }
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  let res: Response;

  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: opts?.signal,
    });
  } catch (err) {
    // network-level error / aborted
    // keep the original error shape
    throw err;
  }

  if (!res.ok) {
    // try to include server message (JSON or text)
    const contentType = res.headers.get("content-type") ?? "";
    let bodyText = "";
    try {
      if (contentType.includes("application/json")) {
        const json = await res.json();
        bodyText = typeof json === "string" ? json : JSON.stringify(json);
      } else {
        bodyText = await res.text();
      }
    } catch (e) {
      bodyText = "[failed to parse server response]";
    }
    throw new Error(`Request failed (${res.status} ${res.statusText}): ${bodyText}`);
  }

  // parse JSON
  try {
    return (await res.json()) as T;
  } catch (err) {
    throw new Error("Response did not contain valid JSON");
  }
}

/**
 * Fetches paginated prediction data from the backend.
 * Returns data with pagination metadata.
 */
export async function fetchPredictionData(
    position: Position | string,
    model: string = "xgb",
    type: string = "predictions",
    limit: number = 50,
    offset: number = 0,
    opts?: { signal?: AbortSignal }
  ): Promise<PaginatedResponse> {
    const payload: FetchDataPayload = {
      position,
      type,
      model,
      limit,
      offset,
    };
    return postJSON<PaginatedResponse>("/api/fetch_data", payload, opts);
  }

/**
 * Wrapper that calls your /get_features endpoint and returns a list of feature names (strings).
 */
export async function fetchFeatures(
  position: Position | string,
  opts?: { signal?: AbortSignal }
): Promise<FeaturesResponse> {
  const payload: FetchFeaturesPayload = { position };
  return postJSON<FeaturesResponse>("/api/get_features", payload, opts);
}