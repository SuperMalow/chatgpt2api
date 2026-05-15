import webConfig from "@/constants/common-env";
import {clearStoredAuthSession, getStoredAuthKey} from "@/store/auth";

type RequestConfig = {
    url?: string;
    method?: string;
    data?: unknown;
    headers?: Record<string, string>;
    responseType?: "json" | "blob" | "text";
    redirectOnUnauthorized?: boolean;
};

type ErrorPayload = {
    detail?: string | { error?: string | { message?: string } };
    error?: string | { message?: string };
    message?: string;
};

function errorMessageFromValue(value: unknown): string {
    if (typeof value === "string") {
        return value;
    }
    if (!value || typeof value !== "object") {
        return "";
    }

    const item = value as { error?: unknown; message?: unknown };
    if (typeof item.message === "string") {
        return item.message;
    }
    return errorMessageFromValue(item.error);
}

const apiBaseUrl = webConfig.apiUrl.replace(/\/$/, "");

function buildUrl(path: string) {
    if (/^https?:\/\//i.test(path)) {
        return path;
    }
    return `${apiBaseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

function isFormData(value: unknown): value is FormData {
    return typeof FormData !== "undefined" && value instanceof FormData;
}

async function parseResponseData<T>(response: Response, responseType: RequestConfig["responseType"]): Promise<T> {
    if (responseType === "blob") {
        return response.blob() as Promise<T>;
    }
    if (responseType === "text") {
        return response.text() as Promise<T>;
    }

    const text = await response.text();
    if (!text) {
        return undefined as T;
    }
    return JSON.parse(text) as T;
}

async function parseErrorPayload(response: Response): Promise<ErrorPayload | null> {
    try {
        return (await response.clone().json()) as ErrorPayload;
    } catch {
        return null;
    }
}

async function requestInternal<T>(config: RequestConfig) {
    const {
        url = "",
        method = "GET",
        data,
        headers: inputHeaders,
        responseType = "json",
        redirectOnUnauthorized = true,
    } = config;
    const authKey = await getStoredAuthKey();
    const headers = {...(inputHeaders || {})};
    if (authKey && !headers.Authorization) {
        headers.Authorization = `Bearer ${authKey}`;
    }

    const init: RequestInit = {
        method,
        headers,
    };
    if (data !== undefined) {
        if (isFormData(data)) {
            init.body = data;
        } else {
            headers["Content-Type"] = headers["Content-Type"] || "application/json";
            init.body = typeof data === "string" ? data : JSON.stringify(data);
        }
    }

    const response = await fetch(buildUrl(url), init);
    if (response.status === 401 && redirectOnUnauthorized && typeof window !== "undefined") {
        if (!window.location.pathname.startsWith("/login")) {
            await clearStoredAuthSession();
            window.location.replace("/login");
            return new Promise<{ data: T }>(() => {});
        }
    }
    if (!response.ok) {
        const payload = await parseErrorPayload(response);
        const message =
            errorMessageFromValue(payload?.detail) ||
            errorMessageFromValue(payload?.error) ||
            payload?.message ||
            response.statusText ||
            `请求失败 (${response.status || 500})`;
        throw new Error(message);
    }

    return {
        data: await parseResponseData<T>(response, responseType),
    };
}

export const request = {
    defaults: {
        baseURL: apiBaseUrl,
    },
    request: requestInternal,
    get<T>(url: string, config: Omit<RequestConfig, "url" | "method" | "data"> = {}) {
        return requestInternal<T>({...config, url, method: "GET"});
    },
    post<T>(url: string, data?: unknown, config: Omit<RequestConfig, "url" | "method" | "data"> = {}) {
        return requestInternal<T>({...config, url, method: "POST", data});
    },
};

type RequestOptions = {
    method?: string;
    body?: unknown;
    headers?: Record<string, string>;
    redirectOnUnauthorized?: boolean;
};

export async function httpRequest<T>(path: string, options: RequestOptions = {}) {
    const {method = "GET", body, headers, redirectOnUnauthorized = true} = options;
    const config: RequestConfig = {
        url: path,
        method,
        data: body,
        headers,
        redirectOnUnauthorized,
    };
    const response = await request.request<T>(config);
    return response.data;
}
