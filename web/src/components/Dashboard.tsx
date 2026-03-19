import { useState, useEffect } from 'react'
import { useAuth } from '../App'
import axios from 'axios'
import { ArrowPathIcon } from '@heroicons/react/24/outline'
import LogViewer from './LogViewer'
import ConfigEditor from './ConfigEditor'
import ServerList from './ServerList'
import PersonaList from './PersonaList'
import PersonaEditor from './PersonaEditor'
import TaskList from './TaskList'
import TaskEditor from './TaskEditor'
import SkillsList from './SkillsList'
import Sidebar from './Sidebar'

const API_BASE = '/api'

interface BotStatus {
  status: string
  online: boolean
  uptime_seconds: number
  started_at: string
  server_count: number
  channel_count: number
  user_name?: string
  user_id?: number
  avatar_url?: string
  status_message?: string
}

const STORAGE_KEY = 'portal_active_tab'

function getInitialTab(): 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills' {
  const saved = localStorage.getItem(STORAGE_KEY)
  // Handle legacy 'status' key - redirect to 'dashboard'
  if (saved === 'dashboard' || saved === 'config' || saved === 'servers' || saved === 'personas' || saved === 'tasks' || saved === 'skills') {
    return saved as 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills'
  }
  if (saved === 'status' || saved === 'logs') {
    return 'dashboard'
  }
  return 'dashboard'
}

interface DashboardProps {
  onOpenServer?: (serverId: string) => void
  onCloseServer?: () => void
}

export default function Dashboard({ onOpenServer, onCloseServer }: DashboardProps) {
  const { token, setToken } = useAuth()
  const [activeTab, setActiveTab] = useState<'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills'>(getInitialTab)
  const [status, setStatus] = useState<BotStatus | null>(null)
  const [loading, setLoading] = useState(true)
  
  // Persona editor state
  const [editingPersona, setEditingPersona] = useState<string | null>(null)  // null = list view, string = editing persona name
  // Task editor state
  // null/undefined = list view, '' = new task, string = editing existing task
  const [editingTask, setEditingTask] = useState<string | null | undefined>(null)
  // Key to force TaskList re-render (reset expanded state) when tab is clicked
  const [tasksKey, setTasksKey] = useState(0)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  const handleTabChange = (tab: 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills') => {
    // Reset to default view when clicking a tab, even if already on that tab
    // This ensures clicking a tab always goes to the list/default view (not previous state)
    // Reset persona editor to list view
    setEditingPersona(null)
    // Reset task editor to list view
    setEditingTask(undefined)
    // Reset TaskList expanded state by changing key
    if (tab === 'tasks') {
      setTasksKey(k => k + 1)
    }
    // Close server drawer if switching away from servers tab
    if (tab !== 'servers' && onCloseServer) {
      onCloseServer()
    }
    setActiveTab(tab)
    localStorage.setItem(STORAGE_KEY, tab)
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/status`, authHeaders)
      setStatus(res.data)
    } catch (err) {
      console.error('Failed to fetch status:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    setToken(null)
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  if (loading) {
    return <div>Loading...</div>
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar 
        activeTab={activeTab} 
        onTabChange={handleTabChange} 
        onLogout={handleLogout} 
      />

      <main style={{ 
        flex: 1, 
        marginLeft: '200px', 
        padding: '1.5rem',
        maxWidth: 'calc(100% - 200px)',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        overflow: 'hidden',
        boxSizing: 'border-box'
      }}>
        {/* Dashboard: Show status + logs as widgets - combined into one view */}
        {activeTab === 'dashboard' && token && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            {/* Status + Quick Actions combined card */}
            <div style={{ 
              backgroundColor: '#1a1a1a', 
              borderRadius: '8px', 
              padding: '1rem',
              flexShrink: 0,
              display: 'flex',
              gap: '1.5rem',
              alignItems: 'center'
            }}>
              {/* Avatar on the left with status indicator */}
              <div style={{ position: 'relative', flexShrink: 0 }}>
                {status?.avatar_url ? (
                  <img 
                    src={status.avatar_url} 
                    alt={`${status.user_name} avatar`}
                    style={{ width: '60px', height: '60px', borderRadius: '50%' }}
                  />
                ) : (
                  <div 
                    style={{ 
                      width: '60px', 
                      height: '60px', 
                      borderRadius: '50%', 
                      backgroundColor: '#333',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '1.5rem'
                    }}
                  >
                    🤖
                  </div>
                )}
                {/* Status indicator dot at bottom-right of avatar */}
                <span 
                  style={{
                    position: 'absolute',
                    bottom: '2px',
                    right: '2px',
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    backgroundColor: status?.online ? '#22c55e' : '#6b7280',
                    border: '2px solid #1a1a1a'
                  }}
                  title={status?.online ? 'Online' : 'Offline'}
                />
              </div>
              
              {/* Info in the middle */}
              <div style={{ flex: 1, display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                {status?.status_message && (
                  <span><strong>Mood:</strong> {status.status_message}</span>
                )}
                <span><strong>Uptime:</strong> {status ? formatUptime(status.uptime_seconds) : 'N/A'}</span>
                <span><strong>Guilds:</strong> {status?.server_count ?? 0}</span>
                <span><strong>Channels:</strong> {status?.channel_count ?? 0}</span>
              </div>

              {/* Quick Actions on the right */}
              <button onClick={fetchStatus} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
                <ArrowPathIcon style={{ width: '1rem', height: '1rem' }} />
                Refresh
              </button>
            </div>
            
            {/* Real-time Logs Widget - fills remaining space */}
            <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <LogViewer token={token} />
            </div>
          </div>
        )}

        {activeTab === 'config' && token && (
          <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <ConfigEditor token={token} />
          </div>
        )}
        {activeTab === 'servers' && token && (
          <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <ServerList token={token} onOpenServer={onOpenServer} />
          </div>
        )}
        {activeTab === 'personas' && token && (
          editingPersona !== null ? (
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
              <PersonaEditor
                token={token}
                personaName={editingPersona || undefined}
                onSave={() => setEditingPersona(null)}
                onSaveWithName={(name) => setEditingPersona(name)}
                onCancel={() => setEditingPersona(null)}
              />
            </div>
          ) : (
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
              <PersonaList
                token={token}
                onSelectPersona={(name) => setEditingPersona(name)}
                onCreateNew={() => setEditingPersona('')}
              />
            </div>
          )
        )}

        {activeTab === 'tasks' && token && (
          editingTask !== undefined && editingTask !== null ? (
            <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
              <TaskEditor
                token={token}
                taskName={editingTask === '' ? undefined : editingTask}
                onSave={() => setEditingTask(undefined)}
                onSaveWithName={(name) => setEditingTask(name || undefined)}
                onCancel={() => setEditingTask(undefined)}
              />
            </div>
          ) : (
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
              <TaskList 
                key={tasksKey}
                token={token} 
                onRequestEdit={(name) => setEditingTask(name)}
                onRequestCreate={() => setEditingTask('')}
              />
            </div>
          )
        )}

        {activeTab === 'skills' && token && (
          <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <SkillsList token={token} />
          </div>
        )}
      </main>
    </div>
  )
}