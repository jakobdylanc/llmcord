import { useState, useEffect } from 'react'
import axios from 'axios'
import { toast } from 'react-hot-toast'
import { ArrowPathIcon, ChevronDownIcon, ChevronRightIcon, PencilSquareIcon, PlusIcon, PlayIcon, ClockIcon } from '@heroicons/react/24/outline'

const API_BASE = '/api'

interface TaskInfo {
  name: string
  filename: string
  enabled: boolean | null
  schedule: string | null
  description: string | null
  status: string
}

interface TaskListProps {
  token: string
  onSelectTask?: (name: string) => void
  onCreateNew?: () => void
  // If provided, show editor inline instead of callbacks
  editingTask?: string | null
  onEditComplete?: () => void
  onRequestEdit?: (name: string) => void
  onRequestCreate?: () => void
}

export default function TaskList({ token, onSelectTask, onCreateNew, onRequestEdit, onRequestCreate }: TaskListProps) {
  const [tasks, setTasks] = useState<TaskInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedTask, setExpandedTask] = useState<string | null>(null)
  const [toggling, setToggling] = useState<string | null>(null)
  const [runningTask, setRunningTask] = useState<string | null>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchTasks()
  }, [])

  const fetchTasks = async () => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/tasks`, authHeaders)
      setTasks(res.data || [])
      setError(null)
    } catch (err: any) {
      console.error('Failed to fetch tasks:', err)
      setError(err.response?.data?.detail || 'Failed to load tasks')
    } finally {
      setLoading(false)
    }
  }

  const toggleTask = async (taskName: string, currentEnabled: boolean) => {
    try {
      setToggling(taskName)
      // Fetch full task config first
      const res = await axios.get(`${API_BASE}/tasks/${taskName}`, authHeaders)
      const config = res.data.config
      config.enabled = !currentEnabled
      
      // Update the task - wrap in { config: ... } for Pydantic model
      await axios.put(`${API_BASE}/tasks/${taskName}`, { config }, authHeaders)
      
      // Refresh the list
      await fetchTasks()
    } catch (err: any) {
      console.error('Failed to toggle task:', err)
      setError(err.response?.data?.detail || 'Failed to toggle task')
    } finally {
      setToggling(null)
    }
  }

  const runTask = async (taskName: string) => {
    let jobId: string | null = null
    
    try {
      setRunningTask(taskName)
      const res = await axios.post(`${API_BASE}/tasks/${taskName}/run`, {}, authHeaders)
      
      if (res.data.success) {
        jobId = res.data.job_id
        toast.success(`Task "${taskName}" queued for execution`)
        
        // Poll for status updates
        const maxAttempts = 20  // 20 * 500ms = 10 seconds max
        let attempts = 0
        
        const pollStatus = async () => {
          if (!jobId) return
          
          try {
            const statusRes = await axios.get(`${API_BASE}/tasks/${taskName}/status`, authHeaders)
            const status = statusRes.data
            
            if (status.status === 'completed') {
              toast.success(`Task "${taskName}" completed successfully`)
            } else if (status.status === 'failed') {
              toast.error(`Task "${taskName}" failed: ${status.error || 'Unknown error'}`)
            }
            // If still queued or running, continue polling
          } catch (err) {
            console.error('Status poll error:', err)
          }
        }
        
        // Start polling
        const pollInterval = setInterval(async () => {
          attempts++
          await pollStatus()
          
          // Get current status to check if we should stop
          try {
            const statusRes = await axios.get(`${API_BASE}/tasks/${taskName}/status`, authHeaders)
            if (statusRes.data.status === 'completed' || statusRes.data.status === 'failed') {
              clearInterval(pollInterval)
              setRunningTask(null)
            } else if (attempts >= maxAttempts) {
              clearInterval(pollInterval)
              setRunningTask(null)
              // Don't show error - just stop polling after timeout
            }
          } catch {
            clearInterval(pollInterval)
            setRunningTask(null)
          }
        }, 500)
        
        // Clean up on unmount
        return () => clearInterval(pollInterval)
      } else {
        toast.error(res.data.message || 'Failed to run task')
      }
    } catch (err: any) {
      console.error('Failed to run task:', err)
      toast.error(err.response?.data?.detail || 'Failed to run task')
    } finally {
      // Only clear if not polling (will be cleared by poll interval)
      if (!jobId) {
        setRunningTask(null)
      }
    }
  }

  const getStatusBadge = (status: string) => {
    const colors: Record<string, string> = {
      scheduled: '#22c55e',  // green
      pending: '#feca57',    // yellow
      running: '#48dbfb',    // blue
      disabled: '#6b7280',   // gray
      error: '#ff6b6b',      // red
      unknown: '#6b7280',
    }
    return (
      <span style={{
        padding: '2px 8px',
        borderRadius: '4px',
        backgroundColor: colors[status] || colors.unknown,
        fontSize: '0.75rem',
        fontWeight: 'bold',
      }}>
        {status.toUpperCase()}
      </span>
    )
  }

  if (loading) {
    return <div>Loading tasks...</div>
  }

  if (error) {
    return (
      <div style={{ padding: '1rem' }}>
        <p style={{ color: '#ff6b6b' }}>{error}</p>
        <button onClick={fetchTasks}>Retry</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ClockIcon style={{ width: '1.5rem', height: '1.5rem' }} />
          Scheduled Tasks
        </h2>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {(onCreateNew || onRequestCreate) && (
            <button onClick={() => { if (onCreateNew) onCreateNew(); if (onRequestCreate) onRequestCreate(); }} style={{ backgroundColor: '#22c55e', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <PlusIcon style={{ width: '1rem', height: '1rem' }} />
              Add New
            </button>
          )}
          <button 
            onClick={fetchTasks} 
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <ArrowPathIcon style={{ width: '1rem', height: '1rem' }} />
            Refresh
          </button>
        </div>
      </div>

      {tasks.length === 0 ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#888' }}>
          <p>No tasks found.</p>
          <p>Create a task file in <code>bot/config/tasks/</code></p>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '1rem' }}>
          {tasks.map((task) => (
            <div
              key={task.name}
              style={{
                border: '1px solid #444',
                borderRadius: '8px',
                backgroundColor: expandedTask === task.name ? '#2a2a2a' : '#1e1e1e',
                overflow: 'hidden',
              }}
            >
              {/* Card Header - Click to expand */}
              <div 
                style={{ 
                  padding: '1rem', 
                  cursor: 'pointer',
                }}
                onClick={() => setExpandedTask(expandedTask === task.name ? null : task.name)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                      {/* Expand icon */}
                      {expandedTask === task.name ? (
                        <ChevronDownIcon style={{ width: '1.25rem', height: '1.25rem', color: '#666' }} />
                      ) : (
                        <ChevronRightIcon style={{ width: '1.25rem', height: '1.25rem', color: '#666' }} />
                      )}
                      <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: '600' }}>
                        {task.name}
                      </h3>
                      {/* Status badge in header */}
                      {getStatusBadge(task.status)}
                    </div>
                    {task.description && (
                      <p style={{ margin: '0 0 0.5rem 1.75rem', color: '#aaa', fontSize: '0.875rem' }}>
                        {task.description}
                      </p>
                    )}
                    {/* Schedule in collapsed view */}
                    {task.schedule && (
                      <div style={{ marginLeft: '1.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#888', fontSize: '0.85rem', fontFamily: 'monospace' }}>
                        <ClockIcon style={{ width: '1rem', height: '1rem' }} />
                        {task.schedule}
                      </div>
                    )}
                  </div>
                  {/* Enable toggle in header */}
                  <div onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => toggleTask(task.name, task.enabled ?? false)}
                      disabled={toggling === task.name}
                      style={{
                        padding: '0.4rem 0.8rem',
                        fontSize: '0.85rem',
                        backgroundColor: (task.enabled ?? false) ? '#22c55e' : '#4b5563',
                        opacity: toggling === task.name ? 0.5 : 1,
                      }}
                    >
                      {(task.enabled ?? false) ? 'ON' : 'OFF'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Expanded content - Configuration and action buttons */}
              {expandedTask === task.name && (
                <div style={{ 
                  padding: '1rem', 
                  backgroundColor: '#252525', 
                  borderTop: '1px solid #444',
                }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#888', fontSize: '0.875rem' }}>
                    Configuration
                  </h4>
                  <pre style={{ 
                    margin: 0, 
                    padding: '1rem', 
                    backgroundColor: '#1a1a1a', 
                    borderRadius: '4px',
                    overflow: 'auto',
                    fontSize: '0.85rem',
                    maxHeight: '300px',
                  }}>
                    {JSON.stringify(tasks.find(t => t.name === task.name), null, 2)}
                  </pre>
                  
                  {/* Action buttons at bottom-left of expanded area */}
                  <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem' }}>
                    {/* Run Now button */}
                    <button 
                      onClick={() => runTask(task.name)}
                      disabled={runningTask === task.name}
                      style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: runningTask === task.name ? 0.5 : 1 }}
                    >
                      <PlayIcon style={{ width: '1rem', height: '1rem' }} />
                      {runningTask === task.name ? 'Running...' : 'Run Now'}
                    </button>
                    
                    {/* Edit button */}
                    {onSelectTask && (
                      <button 
                        onClick={() => onSelectTask(task.name)}
                        style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                      >
                        <PencilSquareIcon style={{ width: '1rem', height: '1rem' }} />
                        Edit
                      </button>
                    )}
                    {onRequestEdit && (
                      <button 
                        onClick={() => onRequestEdit(task.name)}
                        style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                      >
                        <PencilSquareIcon style={{ width: '1rem', height: '1rem' }} />
                        Edit
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}