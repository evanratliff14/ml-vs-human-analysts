"use client";

import { useState, useEffect } from "react";
import {
  fetchPredictionData,
  fetchError,
  fetchPermImportance,
  PaginatedResponse,
  ErrorResponse,
  PermImportanceResponse,
  FeatureImportance,
} from "./fetch";
import { Analytics } from "@vercel/analytics/next"

type Position = "qb" | "rb" | "wr" | "te";

export default function Home() {
  const [selectedPosition, setSelectedPosition] = useState<Position>("qb");
  const [data, setData] = useState<PaginatedResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pageSize] = useState(25); // Fixed to 25 for Vercel free tier
  const [currentOffset, setCurrentOffset] = useState(0);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [errorLoading, setErrorLoading] = useState(false);
  const [errorError, setErrorError] = useState<string | null>(null);
  const [features, setFeatures] = useState<FeatureImportance[] | null>(null);
  const [featuresLoading, setFeaturesLoading] = useState(false);
  const [featuresError, setFeaturesError] = useState<string | null>(null);
  const [showFeatures, setShowFeatures] = useState(false);

  const positions: { value: Position; label: string }[] = [
    { value: "qb", label: "QB" },
    { value: "rb", label: "RB" },
    { value: "wr", label: "WR" },
    { value: "te", label: "TE" },
  ];

  useEffect(() => {
    loadData();
    loadError();
    // Reset features when position changes
    setFeatures(null);
    setShowFeatures(false);
    setFeaturesError(null);
    setErrorError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPosition, currentOffset]);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchPredictionData(
        selectedPosition,
        "xgb",
        "predictions",
        pageSize,
        currentOffset
      );
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const loadError = async () => {
    setErrorLoading(true);
    setErrorError(null);
    try {
      console.log("Loading error for position:", selectedPosition);
      const result: ErrorResponse = await fetchError(selectedPosition);
      console.log("Error result:", result);
      if (result && result.error_text) {
        setErrorText(result.error_text);
        setErrorError(null);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load error data";
      console.error("Error loading error metrics:", err);
      setErrorError(errorMessage);
      setErrorText(null);
    } finally {
      setErrorLoading(false);
    }
  };

  const loadFeatures = async () => {
    if (features && showFeatures) {
      // If already loaded and showing, just toggle
      setShowFeatures(false);
      return;
    }

    setFeaturesLoading(true);
    setFeaturesError(null);
    try {
      console.log("Loading features for position:", selectedPosition);
      const result: PermImportanceResponse = await fetchPermImportance(
        selectedPosition
      );
      console.log("Features result:", result);
      if (result && result.features && Array.isArray(result.features)) {
        setFeatures(result.features);
        setShowFeatures(true);
        setFeaturesError(null);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load features";
      console.error("Error loading features:", err);
      setFeaturesError(errorMessage);
      setFeatures(null);
      setShowFeatures(false);
    } finally {
      setFeaturesLoading(false);
    }
  };

  const handlePositionChange = (position: Position) => {
    setSelectedPosition(position);
    setCurrentOffset(0); // Reset to first page when changing position
  };

  const handlePreviousPage = () => {
    if (currentOffset > 0) {
      setCurrentOffset(Math.max(0, currentOffset - pageSize));
    }
  };

  const handleNextPage = () => {
    if (data && data.has_more) {
      setCurrentOffset(currentOffset + pageSize);
    }
  };

  const currentPage = Math.floor(currentOffset / pageSize) + 1;
  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  // Get all unique column names from the data, excluding headshot_url for special handling
  const columns =
    data && data.data.length > 0
      ? Object.keys(data.data[0])
          .filter((col) => col !== "headshot_url")
          .sort()
      : [];

  // Check if headshot_url exists in data
  const hasHeadshotUrl =
    data && data.data.length > 0 && "headshot_url" in data.data[0];

  // Format cell value for display
  const formatCellValue = (value: unknown): string => {
    if (value === null || value === undefined) return "-";
    if (typeof value === "number") {
      // Format numbers with appropriate precision
      return value.toFixed(2).replace(/\.?0+$/, "");
    }
    return String(value);
  };

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-black sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-semibold mb-2">
            ML vs Human Analysts - Evan Ratliff
          </h1>

          {/* Headshot */}
          <img
            src="/headshot.jpg"
            alt="Evan Ratliff Headshot"
            className="w-24 h-24 rounded-full mb-2"
          />

          <p className="text-gray-400 text-sm">
            Contact me:{" "}
            <a
              href="https://www.linkedin.com/in/ecratliff/"
              className="underline"
            >
              LinkedIn
            </a>{" "}
            -{" "}
            <a
              href="https://github.com/evanratliff14"
              className="underline ml-1"
            >
              GitHub
            </a>
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Position Selector */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-3">
            {positions.map((pos) => (
              <button
                key={pos.value}
                onClick={() => handlePositionChange(pos.value)}
                className={`px-6 py-3 rounded-lg font-semibold transition-all duration-200 ${
                  selectedPosition === pos.value
                    ? "bg-white text-black shadow-lg scale-105"
                    : "bg-gray-900 text-gray-300 hover:bg-gray-800 hover:text-white border border-gray-700"
                }`}
              >
                {pos.label}
              </button>
            ))}
          </div>
        </div>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - Main table (2/3 width on large screens) */}
          <div className="lg:col-span-2">
            {/* Loading State */}
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white"></div>
                <span className="ml-4 text-gray-400">Loading data...</span>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 mb-6">
                <p className="text-red-400 font-medium">Error: {error}</p>
                <button
                  onClick={loadData}
                  className="mt-2 px-4 py-2 bg-red-900 hover:bg-red-800 rounded-md text-sm transition-colors"
                >
                  Retry
                </button>
              </div>
            )}

            {/* Data Table */}
            {!loading && !error && data && (
              <>
                <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden mb-6">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr>
                          {/* Add headshot column header if headshot_url exists */}
                          {hasHeadshotUrl && (
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider bg-gray-800 border-b border-gray-700">
                              Photo
                            </th>
                          )}
                          {columns.map((column) => (
                            <th
                              key={column}
                              className="px-4 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider bg-gray-800 border-b border-gray-700"
                            >
                              {column.replace(/_/g, " ").replace(/\//g, " / ")}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800">
                        {data.data.map((row, rowIndex) => (
                          <tr
                            key={rowIndex}
                            className="hover:bg-gray-800 transition-colors"
                          >
                            {/* Render headshot image if headshot_url exists */}
                            {hasHeadshotUrl && (
                              <td className="px-4 py-3 whitespace-nowrap">
                                {row.headshot_url ? (
                                  <img
                                    src={String(row.headshot_url)}
                                    alt={`${row.player_name || "Player"} headshot`}
                                    className="w-10 h-10 rounded-full object-cover"
                                    onError={(e) => {
                                      // Hide broken images
                                      (e.target as HTMLImageElement).style.display =
                                        "none";
                                    }}
                                  />
                                ) : (
                                  <div className="w-10 h-10 rounded-full bg-gray-700"></div>
                                )}
                              </td>
                            )}
                            {columns.map((column) => (
                              <td
                                key={column}
                                className="px-4 py-3 whitespace-nowrap text-sm text-gray-300"
                              >
                                {formatCellValue(row[column])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Pagination Controls */}
                <div className="flex items-center justify-between bg-gray-900 rounded-lg border border-gray-800 px-6 py-4">
                  <div className="text-sm text-gray-400">
                    Showing {currentOffset + 1} to{" "}
                    {Math.min(currentOffset + pageSize, data.total)} of{" "}
                    {data.total} entries
                  </div>
                  <div className="flex items-center gap-4">
                    <button
                      onClick={handlePreviousPage}
                      disabled={currentOffset === 0 || loading}
                      className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                        currentOffset === 0 || loading
                          ? "bg-gray-800 text-gray-600 cursor-not-allowed"
                          : "bg-gray-800 text-white hover:bg-gray-700 border border-gray-700"
                      }`}
                    >
                      Previous
                    </button>
                    <div className="text-sm text-gray-400">
                      Page {currentPage} of {totalPages}
                    </div>
                    <button
                      onClick={handleNextPage}
                      disabled={!data.has_more || loading}
                      className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                        !data.has_more || loading
                          ? "bg-gray-800 text-gray-600 cursor-not-allowed"
                          : "bg-gray-800 text-white hover:bg-gray-700 border border-gray-700"
                      }`}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </>
            )}

            {/* Empty State */}
            {!loading && !error && data && data.data.length === 0 && (
              <div className="text-center py-12">
                <p className="text-gray-400 text-lg">
                  No data available for this position.
                </p>
              </div>
            )}
          </div>

          {/* Right column - Error and Features (1/3 width on large screens) */}
          <div className="lg:col-span-1 space-y-6">
            {/* Error Display */}
            <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
              <h2 className="text-xl font-semibold mb-4 text-white">
                Model Error Metrics
              </h2>
              {errorLoading ? (
                <div className="flex items-center justify-center py-4">
                  <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-white"></div>
                </div>
              ) : errorError ? (
                <div className="bg-red-900/20 border border-red-700 rounded-lg p-3">
                  <p className="text-red-400 text-sm">{errorError}</p>
                  <button
                    onClick={loadError}
                    className="mt-2 px-3 py-1 bg-red-900 hover:bg-red-800 rounded text-xs transition-colors"
                  >
                    Retry
                  </button>
                </div>
              ) : errorText ? (
                <pre className="text-sm text-gray-300 font-mono whitespace-pre-wrap bg-gray-800 p-4 rounded border border-gray-700">
                  {errorText}
                </pre>
              ) : (
                <p className="text-gray-400 text-sm">No error data available.</p>
              )}
            </div>

            {/* Features Display */}
            <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">
                  Top Features
                </h2>
                <button
                  onClick={loadFeatures}
                  disabled={featuresLoading}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    featuresLoading
                      ? "bg-gray-800 text-gray-600 cursor-not-allowed"
                      : showFeatures
                      ? "bg-gray-700 text-white hover:bg-gray-600 border border-gray-600"
                      : "bg-gray-800 text-white hover:bg-gray-700 border border-gray-700"
                  }`}
                >
                  {featuresLoading
                    ? "Loading..."
                    : showFeatures
                    ? "Hide Features"
                    : "Show Top 15"}
                </button>
              </div>
              {featuresError && (
                <div className="bg-red-900/20 border border-red-700 rounded-lg p-3 mb-4">
                  <p className="text-red-400 text-sm">{featuresError}</p>
                  <button
                    onClick={loadFeatures}
                    className="mt-2 px-3 py-1 bg-red-900 hover:bg-red-800 rounded text-xs transition-colors"
                  >
                    Retry
                  </button>
                </div>
              )}
              {showFeatures && features && features.length > 0 && (
                <div className="space-y-2">
                  {features.map((feature, index) => (
                    <div
                      key={index}
                      className="bg-gray-800 p-3 rounded border border-gray-700"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-white">
                          {index + 1}. {feature.feature.replace(/_/g, " ")}
                        </span>
                        <span className="text-sm text-gray-300">
                          {feature.importance.toFixed(4)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-400">
                        ± {feature.std.toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {showFeatures && !features && !featuresLoading && !featuresError && (
                <p className="text-gray-400 text-sm">
                  No features data available.
                </p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
