import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { ArrowPathIcon } from '@heroicons/react/24/outline'

const API_BASE = '/api'

interface Server {
  id: string  // Use string to avoid JavaScript number precision loss with large Discord guild IDs
  name: string
  icon: string | null
  member_count: number
  channel_count: number
  owner_id: string | null
}

interface ServerListProps {
  token: string
  onOpenServer?: (serverId: string) => void
}

export default function ServerList({ token, onOpenServer }: ServerListProps) {
  const [servers, setServers] = useState<Server[]>([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const handleServerClick = (serverId: string) => {
    if (onOpenServer) {
      onOpenServer(serverId)
    } else {
      navigate(`/servers/${serverId}`)
    }
  }

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchServers()
  }, [])

  const fetchServers = async () => {
    try {
      const res = await axios.get(`${API_BASE}/servers`, authHeaders)
      setServers(res.data.servers || [])
    } catch (err) {
      console.error('Failed to fetch servers:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div>Loading...</div>
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2>Servers</h2>
        <button onClick={fetchServers} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ArrowPathIcon style={{ width: '1rem', height: '1rem' }} />
          Refresh
        </button>
      </div>

      {servers.length === 0 ? (
        <p>No servers found. The bot may not be in any servers yet.</p>
      ) : (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '1rem'
        }}>
          {servers.map((server) => (
            <div 
              key={server.id} 
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                handleServerClick(server.id)
              }}
              style={{ 
                backgroundColor: '#1a1a1a', 
                borderRadius: '8px', 
                padding: '1.5rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#2a2a2a'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#1a1a1a'}
            >
              <div 
                style={{ 
                  width: '64px', 
                  height: '64px', 
                  borderRadius: '50%', 
                  backgroundColor: '#646cff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px',
                  fontWeight: 'bold',
                  marginBottom: '1rem'
                }}
              >
                {server.icon ? (
                  <img 
                    src={server.icon} 
                    alt={server.name}
                    style={{ width: '64px', height: '64px', borderRadius: '50%' }}
                  />
                ) : (
                  server.name.charAt(0).toUpperCase()
                )}
              </div>
              
              <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.2em' }}>{server.name}</h3>
              
              <div style={{ color: '#888', fontSize: '0.9em' }}>
                <p>Members: {server.member_count}</p>
                <p>Channels: {server.channel_count}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}