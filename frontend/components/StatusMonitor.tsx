'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Activity, CheckCircle, XCircle, Loader } from 'lucide-react'

const API_BASE = '/api'

export default function StatusMonitor() {
  const [status, setStatus] = useState<any>(null)

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/status`)
        setStatus(response.data)
      } catch (error) {
        console.error('Error fetching status:', error)
      }
    }, 2000) // Poll every 2 seconds

    return () => clearInterval(interval)
  }, [])

  if (!status || !status.running) {
    return null
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-indigo-600">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Loader className="h-6 w-6 text-indigo-600 animate-spin" />
          <h3 className="text-lg font-semibold text-gray-900">
            Experiment Running
          </h3>
        </div>
        {status.error && (
          <XCircle className="h-6 w-6 text-red-500" />
        )}
      </div>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>{status.current_task}</span>
            <span>{status.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${status.progress}%` }}
            />
          </div>
        </div>

        {status.error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3">
            <p className="text-sm text-red-800">
              <strong>Error:</strong> {status.error}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
