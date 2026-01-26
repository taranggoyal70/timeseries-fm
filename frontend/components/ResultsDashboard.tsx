'use client'

import { TrendingUp, TrendingDown, BarChart, PieChart } from 'lucide-react'
import { BarChart as RechartsBar, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart as RechartsPie, Pie, Cell } from 'recharts'

interface ResultsDashboardProps {
  summary: any
}

const COLORS = ['#10b981', '#ef4444']

export default function ResultsDashboard({ summary }: ResultsDashboardProps) {
  if (!summary) {
    return (
      <div className="bg-white rounded-lg shadow-md p-12 text-center">
        <BarChart className="h-16 w-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          No Results Yet
        </h3>
        <p className="text-gray-600">
          Run an experiment to see results and visualizations
        </p>
      </div>
    )
  }

  const pieData = [
    { name: 'MV Wins', value: summary.mv_wins },
    { name: 'UV Wins', value: summary.uv_wins }
  ]

  const datasetData = Object.entries(summary.by_dataset || {}).map(([name, data]: [string, any]) => ({
    name: name.toUpperCase(),
    'MV Win Rate': data.win_rate,
    'Avg Improvement': data.avg_improvement
  }))

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Experiments</p>
              <p className="text-3xl font-bold text-gray-900">
                {summary.total_experiments}
              </p>
            </div>
            <BarChart className="h-12 w-12 text-indigo-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">MV Win Rate</p>
              <p className="text-3xl font-bold text-green-600">
                {summary.mv_win_rate.toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="h-12 w-12 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg MAPE Improvement</p>
              <p className={`text-3xl font-bold ${summary.avg_mape_improvement > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {summary.avg_mape_improvement > 0 ? '+' : ''}{summary.avg_mape_improvement.toFixed(2)}%
              </p>
            </div>
            {summary.avg_mape_improvement > 0 ? (
              <TrendingUp className="h-12 w-12 text-green-600" />
            ) : (
              <TrendingDown className="h-12 w-12 text-red-600" />
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Datasets Tested</p>
              <p className="text-3xl font-bold text-gray-900">
                {summary.datasets?.length || 0}
              </p>
            </div>
            <PieChart className="h-12 w-12 text-indigo-600" />
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Win Rate Pie Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            MV vs UV Win Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RechartsPie>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </RechartsPie>
          </ResponsiveContainer>
        </div>

        {/* Dataset Performance */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Performance by Dataset
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RechartsBar data={datasetData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="MV Win Rate" fill="#10b981" />
              <Bar dataKey="Avg Improvement" fill="#3b82f6" />
            </RechartsBar>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Stats */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Dataset Details
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Dataset
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Total Experiments
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  MV Wins
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Win Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Avg Improvement
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {Object.entries(summary.by_dataset || {}).map(([name, data]: [string, any]) => (
                <tr key={name}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {name.toUpperCase()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {data.total}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {data.mv_wins}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      data.win_rate > 50 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {data.win_rate.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={data.avg_improvement > 0 ? 'text-green-600' : 'text-red-600'}>
                      {data.avg_improvement > 0 ? '+' : ''}{data.avg_improvement.toFixed(2)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
