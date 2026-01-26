'use client'

import { useState } from 'react'
import axios from 'axios'
import { Play, Settings } from 'lucide-react'

const API_BASE = '/api'

interface ExperimentFormProps {
  onComplete: () => void
}

export default function ExperimentForm({ onComplete }: ExperimentFormProps) {
  const [formData, setFormData] = useState({
    dataset: 'stocks',
    model_size: 'base',
    device: 'cuda',
    alpha_values: [1.0],
    forecast_horizons: [21],
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    step_months: 3,
    quick_test: true
  })

  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      await axios.post(`${API_BASE}/experiment`, formData)
      alert('Experiment started! Check the status monitor above.')
      setTimeout(onComplete, 5000) // Refresh after 5 seconds
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Error starting experiment')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center space-x-3 mb-6">
        <Settings className="h-6 w-6 text-indigo-600" />
        <h2 className="text-2xl font-bold text-gray-900">
          Configure Experiment
        </h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Dataset Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Dataset
          </label>
          <select
            value={formData.dataset}
            onChange={(e) => setFormData({ ...formData, dataset: e.target.value })}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-gray-900 bg-white"
          >
            <option value="stocks">Magnificent-7 Stocks (K=7)</option>
            <option value="rates">FRED Interest Rates (K=10)</option>
            <option value="combined">Combined Dataset (K=17)</option>
          </select>
        </div>

        {/* Model Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model Size
            </label>
            <select
              value={formData.model_size}
              onChange={(e) => setFormData({ ...formData, model_size: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 text-gray-900 bg-white"
            >
              <option value="small">Small (fastest)</option>
              <option value="base">Base (recommended)</option>
              <option value="large">Large (most accurate)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Device
            </label>
            <select
              value={formData.device}
              onChange={(e) => setFormData({ ...formData, device: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 text-gray-900 bg-white"
            >
              <option value="cuda">GPU (CUDA)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
        </div>

        {/* Date Range */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Start Date
            </label>
            <input
              type="date"
              value={formData.start_date}
              onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 text-gray-900 bg-white"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              End Date
            </label>
            <input
              type="date"
              value={formData.end_date}
              onChange={(e) => setFormData({ ...formData, end_date: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 text-gray-900 bg-white"
            />
          </div>
        </div>

        {/* Quick Test Toggle */}
        <div className="flex items-center space-x-3">
          <input
            type="checkbox"
            id="quick_test"
            checked={formData.quick_test}
            onChange={(e) => setFormData({ ...formData, quick_test: e.target.checked })}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="quick_test" className="text-sm font-medium text-gray-700">
            Quick Test Mode (limited parameters, faster execution)
          </label>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 disabled:bg-gray-400 transition-colors"
        >
          <Play className="h-5 w-5" />
          <span>{loading ? 'Starting...' : 'Run Experiment'}</span>
        </button>
      </form>
    </div>
  )
}
