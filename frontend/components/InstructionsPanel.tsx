'use client'

import { Terminal, Play, Download, BarChart } from 'lucide-react'

export default function InstructionsPanel() {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-indigo-600 rounded-lg p-6 mb-6">
      <div className="flex items-start space-x-3">
        <Terminal className="h-6 w-6 text-indigo-600 mt-1 flex-shrink-0" />
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            📋 How to Use This Application
          </h3>
          
          <div className="space-y-4 text-sm">
            {/* Step 1 */}
            <div className="bg-white rounded-md p-4 shadow-sm">
              <div className="flex items-center space-x-2 mb-2">
                <Download className="h-5 w-5 text-green-600" />
                <h4 className="font-semibold text-gray-900">Step 1: Download Data</h4>
              </div>
              <p className="text-gray-700 mb-2">
                Open a terminal and run:
              </p>
              <code className="block bg-gray-900 text-green-400 p-3 rounded text-xs font-mono">
                cd /Users/tarang/CascadeProjects/windsurf-project/chronos2-forecasting<br/>
                python main.py --download-only --dataset stocks
              </code>
              <p className="text-gray-600 mt-2 text-xs">
                ⏱️ Takes 1-2 minutes. Downloads Magnificent-7 stock data from Yahoo Finance.
              </p>
            </div>

            {/* Step 2 */}
            <div className="bg-white rounded-md p-4 shadow-sm">
              <div className="flex items-center space-x-2 mb-2">
                <Play className="h-5 w-5 text-blue-600" />
                <h4 className="font-semibold text-gray-900">Step 2: Run Experiment</h4>
              </div>
              <p className="text-gray-700 mb-2">
                Choose one option:
              </p>
              
              <div className="space-y-2">
                <div>
                  <p className="text-gray-800 font-medium text-xs mb-1">🚀 Quick Test (Recommended):</p>
                  <code className="block bg-gray-900 text-green-400 p-3 rounded text-xs font-mono">
                    python main.py --quick-test --dataset stocks --device cpu
                  </code>
                  <p className="text-gray-600 mt-1 text-xs">⏱️ 10-30 minutes</p>
                </div>
                
                <div>
                  <p className="text-gray-800 font-medium text-xs mb-1">⚡ Single Example (Fastest):</p>
                  <code className="block bg-gray-900 text-green-400 p-3 rounded text-xs font-mono">
                    python example_single_forecast.py
                  </code>
                  <p className="text-gray-600 mt-1 text-xs">⏱️ 2-5 minutes</p>
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="bg-white rounded-md p-4 shadow-sm">
              <div className="flex items-center space-x-2 mb-2">
                <BarChart className="h-5 w-5 text-purple-600" />
                <h4 className="font-semibold text-gray-900">Step 3: View Results</h4>
              </div>
              <p className="text-gray-700">
                When experiments complete, click the <strong>"View Results"</strong> tab above to see:
              </p>
              <ul className="list-disc list-inside text-gray-600 mt-2 space-y-1 text-xs">
                <li>Summary statistics</li>
                <li>MV vs UV comparison charts</li>
                <li>Performance by dataset</li>
                <li>Detailed metrics tables</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
            <p className="text-xs text-yellow-800">
              <strong>💡 Note:</strong> The web UI is for viewing results. Run experiments via terminal for best performance and progress tracking.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
