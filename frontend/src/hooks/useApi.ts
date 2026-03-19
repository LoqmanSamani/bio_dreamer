/**
 * useApi — Generic API Request Hook.
 *
 * Purpose:
 *   Custom React hook that wraps fetch calls to the BioDreamer backend
 *   API. Provides consistent error handling, loading states, and
 *   automatic JSON parsing.
 *
 * Returns:
 *   { data, error, isLoading, mutate, refetch }
 *
 * Features:
 *   - GET requests with SWR-style caching and revalidation
 *   - POST/PUT/DELETE via mutate() function
 *   - Automatic base URL from environment config
 *   - Error normalisation: network errors, HTTP errors, API error responses
 *   - Request cancellation via AbortController on unmount
 *   - Optional polling interval for long-running job status
 *   - Type-safe generics: useApi<ResponseType>(url, options)
 *
 * Example:
 *   const { data, isLoading } = useApi<HealthResponse>("/api/health");
 *   const { mutate: submitJob } = useApi<JobResponse>("/api/protein-dreamer/submit", { method: "POST" });
 */
