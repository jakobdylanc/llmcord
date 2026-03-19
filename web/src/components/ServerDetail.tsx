import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'

const API_BASE = '/api'

interface GuildMember {
  id: number
  username: string
  display_name: string | null
  is_owner: boolean
}

interface GuildChannel {
  id: number
  name: string
  type: string
}

interface GuildPermissions {
  can_send_messages: boolean
  can_embed_links: boolean
  can_attach_files: boolean
  can_use_external_emojis: boolean
  can_manage_messages: boolean
  can_manage_channels: boolean
  can_kick_members: boolean
  can_ban_members: boolean
  can_manage_guild: boolean
}

interface ServerDetail {
  id: number
  name: string
  icon: string | null
  owner_id: number | null
  member_count: number
  channel_count: number
  members: GuildMember[]
  channels: GuildChannel[]
  permissions: GuildPermissions
}

interface ServerDetailProps {
  token: string
}

export default function ServerDetail({ token }: ServerDetailProps) {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [server, setServer] = useState<ServerDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'members' | 'channels' | 'permissions'>('members')
  const [permissionMessage, setPermissionMessage] = useState<string | null>(null)

  console.log('ServerDetail: token=', token ? 'exists' : 'NULL', 'id=', id)

  useEffect(() => {
    console.log('useEffect triggered: id=', id, 'token=', token ? 'exists' : 'NULL')
    if (id && token) {
      // Keep as string to avoid JavaScript number precision loss with large guild IDs
      fetchServerDetail(id)
    } else if (!token) {
      console.error('No token in useEffect!')
      setError('Not authenticated. Please login again.')
      setLoading(false)
    }
  }, [id, token])

  const fetchServerDetail = async (guildId: string) => {
    if (!token) {
      console.error('No token available!')
      setError('Not authenticated. Please login again.')
      setLoading(false)
      return
    }
    
    try {
      console.log('Fetching server detail for guildId:', guildId, '(type:', typeof guildId + ')')
      console.log('Using token:', token.substring(0, 20) + '...')
      
      const res = await axios.get(`${API_BASE}/servers/${guildId}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      console.log('Server detail response:', res.data)
      setServer(res.data)
    } catch (err: any) {
      console.error('Failed to fetch server detail:', err)
      console.error('Error status:', err.response?.status)
      console.error('Error response:', err.response?.data)
      console.error('Error message:', err.message)
      
      if (err.response?.status === 404) {
        setError('Server not found (404). The bot may not be in this server.')
      } else if (err.response?.status === 401) {
        setError('Unauthorized. Please login again.')
      } else if (err.response?.status === 403) {
        setError('Forbidden. You do not have access to this server.')
      } else {
        setError(err.response?.data?.detail || err.message || 'Failed to load server details')
      }
    } finally {
      setLoading(false)
    }
  }

  const updatePermission = async (permission: string, action: 'grant' | 'revoke') => {
    if (!id) return
    
    setPermissionMessage(null)
    try {
      const res = await axios.put(
        `${API_BASE}/servers/${id}/permissions`,
        { action, permission },
        { headers: { Authorization: `Bearer ${token}` } }
      )
      setPermissionMessage(res.data.message)
      // Refresh permissions - keep as string to avoid precision loss
      if (id) {
        fetchServerDetail(id)
      }
    } catch (err: any) {
      setPermissionMessage(err.response?.data?.message || 'Failed to update permission')
    }
  }

  if (loading) {
    return <div>Loading...</div>
  }

  if (error) {
    return (
      <div>
      <button onClick={() => navigate('/')}>&larr; Back to Dashboard</button>
        <p style={{ color: 'red' }}>{error}</p>
      </div>
    )
  }

  if (!server) {
    return (
      <div>
        <button onClick={() => navigate('/')}>&larr; Back to Dashboard</button>
        <p>Server not found</p>
      </div>
    )
  }

  const permissionList = [
    { key: 'can_send_messages', label: 'Send Messages', discord: 'send_messages' },
    { key: 'can_embed_links', label: 'Embed Links', discord: 'embed_links' },
    { key: 'can_attach_files', label: 'Attach Files', discord: 'attach_files' },
    { key: 'can_use_external_emojis', label: 'External Emoji', discord: 'external_emojis' },
    { key: 'can_manage_messages', label: 'Manage Messages', discord: 'manage_messages' },
    { key: 'can_manage_channels', label: 'Manage Channels', discord: 'manage_channels' },
    { key: 'can_kick_members', label: 'Kick Members', discord: 'kick_members' },
    { key: 'can_ban_members', label: 'Ban Members', discord: 'ban_members' },
    { key: 'can_manage_guild', label: 'Manage Server', discord: 'manage_guild' },
  ]

  return (
    <div>
      <button onClick={() => navigate('/')}>&larr; Back to Dashboard</button>
      
      {/* Server Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ position: 'relative' }}>
          {server.icon ? (
            <img 
              src={server.icon} 
              alt={server.name}
              style={{ width: '80px', height: '80px', borderRadius: '50%' }}
            />
          ) : (
            <div style={{ 
              width: '80px', 
              height: '80px', 
              borderRadius: '50%', 
              backgroundColor: '#646cff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '2rem',
              fontWeight: 'bold'
            }}>
              {server.name.charAt(0).toUpperCase()}
            </div>
          )}
        </div>
        <div>
          <h2 style={{ margin: 0 }}>{server.name}</h2>
          <p style={{ margin: '0.25rem 0', color: '#888' }}>
            {server.member_count} members • {server.channel_count} channels
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
        <button 
          onClick={() => setActiveTab('members')}
          style={{ backgroundColor: activeTab === 'members' ? '#646cff' : '#1a1a1a' }}
        >
          Members ({server.members.length})
        </button>
        <button 
          onClick={() => setActiveTab('channels')}
          style={{ backgroundColor: activeTab === 'channels' ? '#646cff' : '#1a1a1a' }}
        >
          Channels ({server.channels.length})
        </button>
        <button 
          onClick={() => setActiveTab('permissions')}
          style={{ backgroundColor: activeTab === 'permissions' ? '#646cff' : '#1a1a1a' }}
        >
          Permissions
        </button>
      </div>

      {/* Members Tab */}
      {activeTab === 'members' && (
        <div style={{ 
          maxHeight: '400px', 
          overflow: 'auto', 
          backgroundColor: '#1a1a1a', 
          borderRadius: '8px',
          padding: '1rem'
        }}>
          {server.members.length === 0 ? (
            <p>No members found</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {server.members.slice(0, 100).map((member) => (
                <div 
                  key={member.id}
                  style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    padding: '0.5rem',
                    backgroundColor: '#2a2a2a',
                    borderRadius: '4px'
                  }}
                >
                  <div>
                    <span style={{ fontWeight: 'bold' }}>{member.username}</span>
                    {member.display_name && (
                      <span style={{ color: '#888', marginLeft: '0.5rem' }}>
                        (aka {member.display_name})
                      </span>
                    )}
                    {member.is_owner && (
                      <span style={{ 
                        marginLeft: '0.5rem', 
                        padding: '0.125rem 0.375rem',
                        backgroundColor: '#f59e0b',
                        borderRadius: '4px',
                        fontSize: '0.75rem'
                      }}>
                        Owner
                      </span>
                    )}
                  </div>
                  <span style={{ color: '#666', fontSize: '0.875rem' }}>ID: {member.id}</span>
                </div>
              ))}
              {server.members.length > 100 && (
                <p style={{ color: '#888', textAlign: 'center' }}>
                  ... and {server.members.length - 100} more members
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Channels Tab */}
      {activeTab === 'channels' && (
        <div style={{ 
          maxHeight: '400px', 
          overflow: 'auto', 
          backgroundColor: '#1a1a1a', 
          borderRadius: '8px',
          padding: '1rem'
        }}>
          {server.channels.length === 0 ? (
            <p>No channels found</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {server.channels.map((channel) => (
                <div 
                  key={channel.id}
                  style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    padding: '0.5rem',
                    backgroundColor: '#2a2a2a',
                    borderRadius: '4px'
                  }}
                >
                  <div>
                    <span style={{ fontWeight: 'bold' }}># {channel.name}</span>
                    <span style={{ 
                      marginLeft: '0.5rem', 
                      padding: '0.125rem 0.375rem',
                      backgroundColor: '#374151',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      color: '#9ca3af'
                    }}>
                      {channel.type}
                    </span>
                  </div>
                  <span style={{ color: '#666', fontSize: '0.875rem' }}>ID: {channel.id}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Permissions Tab */}
      {activeTab === 'permissions' && (
        <div style={{ 
          backgroundColor: '#1a1a1a', 
          borderRadius: '8px',
          padding: '1rem'
        }}>
          <h3 style={{ marginTop: 0 }}>Bot Permissions</h3>
          
          {permissionMessage && (
            <div style={{ 
              padding: '0.75rem', 
              marginBottom: '1rem',
              backgroundColor: permissionMessage.includes('Error') || permissionMessage.includes('Invalid') || permissionMessage.includes('Failed') ? '#7f1d1d' : '#14532d',
              borderRadius: '4px'
            }}>
              {permissionMessage}
            </div>
          )}
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {permissionList.map((perm) => {
              const hasPermission = server.permissions[perm.key as keyof GuildPermissions]
              return (
                <div 
                  key={perm.key}
                  style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    padding: '0.75rem',
                    backgroundColor: '#2a2a2a',
                    borderRadius: '4px'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span 
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: hasPermission ? '#22c55e' : '#6b7280'
                      }}
                    />
                    <span>{perm.label}</span>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={() => updatePermission(perm.discord, 'grant')}
                      style={{ 
                        padding: '0.25rem 0.5rem', 
                        fontSize: '0.75rem',
                        backgroundColor: '#22c55e',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Grant
                    </button>
                    <button
                      onClick={() => updatePermission(perm.discord, 'revoke')}
                      style={{ 
                        padding: '0.25rem 0.5rem', 
                        fontSize: '0.75rem',
                        backgroundColor: '#ef4444',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Revoke
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
          
          <p style={{ color: '#666', fontSize: '0.875rem', marginTop: '1rem' }}>
            Note: Permission changes are simulated. Actual Discord permission management requires administrator access.
          </p>
        </div>
      )}
    </div>
  )
}