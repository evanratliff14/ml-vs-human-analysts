"use client";

import { useState, useEffect } from "react";
import { fetchPredictionData, PaginatedResponse } from "./fetch";

type Position = "qb" | "rb" | "wr" | "te";

export default function Home() {
  const [selectedPosition, setSelectedPosition] = useState<Position>("qb");
  const [data, setData] = useState<PaginatedResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pageSize, setPageSize] = useState(50);
  const [currentOffset, setCurrentOffset] = useState(0);

  const positions: { value: Position; label: string }[] = [
    { value: "qb", label: "QB" },
    { value: "rb", label: "RB" },
    { value: "wr", label: "WR" },
    { value: "te", label: "TE" },
  ];

  const pageSizes = [25, 50, 100];

  useEffect(() => {
    loadData();
  }, [selectedPosition, pageSize, currentOffset]);

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

  const handlePositionChange = (position: Position) => {
    setSelectedPosition(position);
    setCurrentOffset(0); // Reset to first page when changing position
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
    setCurrentOffset(0); // Reset to first page when changing page size
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

  // Get all unique column names from the data
  const columns =
    data && data.data.length > 0
      ? Object.keys(data.data[0]).sort()
      : [];

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
          <h1 className="text-3xl font-bold mb-2">ML vs Human Analysts - by Evan Ratliff</h1>
          <p className="text-gray-400 text-sm">Contact me: https://www.linkedin.com/in/ecratliff/ - https://github.com/evanratliff14</p>

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

        {/* Page Size Selector */}
        <div className="mb-6 flex items-center gap-4">
          <label className="text-sm text-gray-400">Rows per page:</label>
          <div className="flex gap-2">
            {pageSizes.map((size) => (
              <button
                key={size}
                onClick={() => handlePageSizeChange(size)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  pageSize === size
                    ? "bg-gray-800 text-white border border-gray-600"
                    : "bg-gray-900 text-gray-400 hover:bg-gray-800 hover:text-gray-300 border border-gray-700"
                }`}
              >
                {size}
              </button>
            ))}
          </div>
        </div>

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
                {Math.min(currentOffset + pageSize, data.total)} of {data.total}{" "}
                entries
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
            <p className="text-gray-400 text-lg">No data available for this position.</p>
          </div>
        )}
      </main>
    </div>
  );
}
