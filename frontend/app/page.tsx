'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Activity, TrendingUp, Database, Play, Download, BarChart3 } from 'lucide-react'
import ExperimentForm from '@/components/ExperimentForm'
import ResultsDashboard from '@/components/ResultsDashboard'
import StatusMonitor from '@/components/StatusMonitor'
import InstructionsPanel from '@/components/InstructionsPanel'

const API_BASE = '/api'

export default function Home() {
  const [activeTab, setActiveTab] = useState('experiment')
  const [datasets, setDatasets] = useState<any>(null)
  const [summary, setSummary] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    checkDatasets()
    loadSummary()
  }, [])

  const checkDatasets = async () => {
    try {
      const response = await axios.get(`${API_BASE}/datasets`)
      setDatasets(response.data.datasets)
    } catch (error) {
      console.error('Error checking datasets:', error)
    }
  }

  const loadSummary = async () => {
    try {
      const response = await axios.get(`${API_BASE}/summary`)
      setSummary(response.data)
    } catch (error) {
      console.log('No results yet')
    }
  }

  const downloadDataset = async (dataset: string) => {
    setLoading(true)
    try {
      const response = await axios.post(`${API_BASE}/download/${dataset}`)
      await checkDatasets()
      
      if (response.data.status === 'success') {
        alert(`✓ ${dataset} dataset downloaded successfully!\n\nYou can now run experiments with this data.`)
      } else if (response.data.message) {
        alert(`${response.data.message}\n\nNote: Download happens in background. This may take 1-2 minutes.`)
      }
    } catch (error: any) {
      alert(`Error downloading ${dataset}: ${error.response?.data?.detail || error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="h-8 w-8 text-indigo-600" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  Chronos-2 Forecasting
                </h1>
                <p className="text-sm text-gray-600">
                  Multivariate vs Univariate Time Series Analysis
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {summary && (
                <div className="text-right">
                  <p className="text-sm text-gray-600">Last Run</p>
                  <p className="text-lg font-semibold text-indigo-600">
                    {summary.total_experiments} experiments
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="bg-white rounded-lg shadow-md p-2 flex space-x-2">
          <button
            onClick={() => setActiveTab('experiment')}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-md transition-colors ${
              activeTab === 'experiment'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Play className="h-5 w-5" />
            <span className="font-medium">Run Experiment</span>
          </button>
          <button
            onClick={() => setActiveTab('results')}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-md transition-colors ${
              activeTab === 'results'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <BarChart3 className="h-5 w-5" />
            <span className="font-medium">View Results</span>
          </button>
          <button
            onClick={() => setActiveTab('data')}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-md transition-colors ${
              activeTab === 'data'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Database className="h-5 w-5" />
            <span className="font-medium">Manage Data</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {activeTab === 'experiment' && (
          <div className="space-y-6">
            <InstructionsPanel />
            <StatusMonitor />
            <ExperimentForm onComplete={loadSummary} />
          </div>
        )}

        {activeTab === 'results' && (
          <ResultsDashboard summary={summary} />
        )}

        {activeTab === 'data' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              Dataset Management
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Stocks */}
              <div className="border-2 border-gray-200 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Magnificent-7 Stocks (K=7)
                  </h3>
                  {datasets?.stocks && (
                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded">
                      Downloaded
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600 mb-2">
                  <strong>Tickers:</strong> AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
                </p>
                <p className="text-sm text-gray-500 mb-4">
                  Source: Yahoo Finance
                </p>
                <button
                  onClick={() => downloadDataset('stocks')}
                  disabled={loading}
                  className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400"
                >
                  <Download className="h-4 w-4" />
                  <span>{datasets?.stocks ? 'Re-download' : 'Download'}</span>
                </button>
              </div>

              {/* Interest Rates */}
              <div className="border-2 border-gray-200 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Treasury Interest Rates (K=10)
                  </h3>
                  {datasets?.rates && (
                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded">
                      Downloaded
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600 mb-2">
                  <strong>Maturities:</strong> 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
                </p>
                <p className="text-sm text-gray-500 mb-4">
                  Source: FRED
                </p>
                <button
                  onClick={() => downloadDataset('rates')}
                  disabled={loading}
                  className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400"
                >
                  <Download className="h-4 w-4" />
                  <span>{datasets?.rates ? 'Re-download' : 'Download'}</span>
                </button>
              </div>

              {/* Combined */}
              <div className="border-2 border-gray-200 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Combined Dataset (K=17)
                  </h3>
                  {datasets?.combined && (
                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded">
                      Downloaded
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600 mb-2">
                  <strong>Both stocks and rates together</strong>
                </p>
                <p className="text-sm text-gray-500 mb-4">
                  7 stocks + 10 interest rates = 17 series
                </p>
                <button
                  onClick={() => downloadDataset('combined')}
                  disabled={loading}
                  className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400"
                >
                  <Download className="h-4 w-4" />
                  <span>{datasets?.combined ? 'Re-download' : 'Download'}</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
