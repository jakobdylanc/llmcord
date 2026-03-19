import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API_BASE = '/api'

interface LogEntry {
  id: number
  timestamp: string
  level: string
  message: string
  logger: string
}

interface LogViewerProps {
  token: string
}

export default function LogViewer({ token }: LogViewerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [levelFilter, setLevelFilter] = useState<string>('ALL')
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Fetch initial logs
  useEffect(() => {
    fetchLogs()
  }, [])

  const fetchLogs = async () => {
    try {
      const params = levelFilter !== 'ALL' ? { level: levelFilter } : {}
      const res = await axios.get(`${API_BASE}/logs`, { ...authHeaders, params })
      setLogs(res.data.logs || [])
    } catch (err) {
      console.error('Failed to fetch logs:', err)
    }
  }

  // Connect to WebSocket for real-time logs
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/logs`
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
      setWsConnected(true)
    }

    ws.onmessage = (event) => {
      if (event.data === 'pong') return // Ignore ping responses
      
      try {
        const log = JSON.parse(event.data)
        setLogs(prev => [log, ...prev].slice(0, 500)) // Keep last 500 logs
      } catch (err) {
        console.error('Failed to parse log:', err)
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setWsConnected(false)
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
    }

    return () => {
      ws.close()
    }
  }, [])

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleLevelChange = (level: string) => {
    setLevelFilter(level)
    fetchLogs()
  }

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return '#ff6b6b'
      case 'WARNING': return '#feca57'
      case 'INFO': return '#48dbfb'
      case 'DEBUG': return '#a29bfe'
      default: return '#fff'
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
      {/* Header row with title, status badge, and filter */}
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '0.5rem', flexShrink: 0 }}>
        <span style={{ fontWeight: 'bold', fontSize: '1.1em' }}>Logs</span>
        <span style={{ 
          padding: '2px 6px', 
          borderRadius: '4px',
          backgroundColor: wsConnected ? '#10ac84' : '#ee5253',
          fontSize: '0.7em'
        }}>
          {wsConnected ? 'Live' : 'Disconnected'}
        </span>
        <span style={{ marginLeft: 'auto' }}>
          <label>Filter: </label>
          <select 
            value={levelFilter} 
            onChange={(e) => handleLevelChange(e.target.value)}
            style={{ padding: '0.25em', borderRadius: '4px', marginLeft: '0.25rem' }}
          >
            <option value="ALL">All</option>
            <option value="DEBUG">Debug</option>
            <option value="INFO">Info</option>
            <option value="WARNING">Warning</option>
            <option value="ERROR">Error</option>
          </select>
        </span>
      </div>

      {/* Log content - fills remaining space */}
      <div style={{ 
        backgroundColor: '#1a1a1a', 
        borderRadius: '8px', 
        padding: '0.5rem',
        flex: 1,
        overflowY: 'auto', 
        fontFamily: 'monospace', 
        fontSize: '0.85em',
        minHeight: 0
      }}>
        {logs.map((log) => (
          <div key={log.id} style={{ marginBottom: '0.25rem' }}>
            <span style={{ color: '#888', marginRight: '0.5rem' }}>
              {new Date(log.timestamp).toLocaleTimeString()}
            </span>
            <span style={{ 
              color: getLevelColor(log.level), 
              fontWeight: 'bold',
              marginRight: '0.5rem',
              minWidth: '60px',
              display: 'inline-block'
            }}>
              [{log.level}]
            </span>
            <span style={{ color: '#ddd' }}>{log.message}</span>
          </div>
        ))}
        <div ref={logsEndRef} />
        {logs.length === 0 && <p>No logs available</p>}
      </div>
    </div>
  )
}