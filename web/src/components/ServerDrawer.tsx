import { useState, useEffect } from 'react'
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

interface ServerDrawerProps {
  serverId: string | null
  token: string
  onClose: () => void
}

export default function ServerDrawer({ serverId, token, onClose }: ServerDrawerProps) {
  const [server, setServer] = useState<ServerDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'members' | 'channels' | 'permissions'>('members')
  const [permissionMessage, setPermissionMessage] = useState<string | null>(null)

  useEffect(() => {
    if (serverId && token) {
      fetchServerDetail(serverId)
    }
  }, [serverId, token])

  const fetchServerDetail = async (guildId: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.get(`${API_BASE}/servers/${guildId}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      setServer(res.data)
    } catch (err: any) {
      if (err.response?.status === 404) {
        setError('Server not found (404). The bot may not be in this server.')
      } else {
        setError('Failed to load server details.')
      }
    } finally {
      setLoading(false)
    }
  }

  const updatePermission = async (permission: string, granted: boolean) => {
    if (!serverId || !server) return
    
    try {
      const res = await axios.put(
        `${API_BASE}/servers/${serverId}/permissions`,
        { [permission]: granted },
        { headers: { Authorization: `Bearer ${token}` } }
      )
      setServer({ ...server, permissions: res.data.permissions })
      setPermissionMessage(`${permission} ${granted ? 'granted' : 'revoked'} successfully`)
      setTimeout(() => setPermissionMessage(null), 3000)
    } catch (err: any) {
      setPermissionMessage(`Error: ${err.response?.data?.detail || 'Failed to update permission'}`)
      setTimeout(() => setPermissionMessage(null), 3000)
    }
  }

  if (!serverId) return null

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          zIndex: 999,
        }}
      />

      {/* Drawer */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          width: '500px',
          maxWidth: '100%',
          height: '100vh',
          backgroundColor: '#1a1a1a',
          boxShadow: '-4px 0 20px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '1rem',
            borderBottom: '1px solid #333',
          }}
        >
          <h2 style={{ margin: 0, fontSize: '1.2rem' }}>
            {server?.name || (loading ? 'Loading...' : 'Server Details')}
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#fff',
              fontSize: '1.5rem',
              cursor: 'pointer',
              padding: '0.25rem 0.5rem',
            }}
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'auto', padding: '1rem' }}>
          {loading && <div>Loading...</div>}
          
          {error && (
            <div style={{ color: '#ff6b6b', padding: '1rem' }}>
              {error}
            </div>
          )}

          {server && !loading && !error && (
            <>
              {/* Server Info */}
              <div style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                {server.icon ? (
                  <img
                    src={server.icon}
                    alt={server.name}
                    style={{ width: '48px', height: '48px', borderRadius: '50%' }}
                  />
                ) : (
                  <div
                    style={{
                      width: '48px',
                      height: '48px',
                      borderRadius: '50%',
                      backgroundColor: '#646cff',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '1.5rem',
                    }}
                  >
                    {server.name.charAt(0).toUpperCase()}
                  </div>
                )}
                <div>
                  <p style={{ margin: 0 }}><strong>Members:</strong> {server.member_count}</p>
                  <p style={{ margin: 0 }}><strong>Channels:</strong> {server.channel_count}</p>
                </div>
              </div>

              {/* Tabs */}
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                {(['members', 'channels', 'permissions'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    style={{
                      padding: '0.5rem 1rem',
                      backgroundColor: activeTab === tab ? '#646cff' : '#333',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer',
                      textTransform: 'capitalize',
                    }}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              {activeTab === 'members' && (
                <div style={{ maxHeight: '300px', overflow: 'auto' }}>
                  {server.members.length === 0 ? (
                    <p>No members found</p>
                  ) : (
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      {server.members.slice(0, 50).map((member) => (
                        <li
                          key={member.id}
                          style={{
                            padding: '0.5rem',
                            borderBottom: '1px solid #333',
                            display: 'flex',
                            justifyContent: 'space-between',
                          }}
                        >
                          <span>
                            {member.display_name || member.username}
                            {member.is_owner && <span style={{ color: '#ffd700' }}> 👑</span>}
                          </span>
                          <span style={{ color: '#888', fontSize: '0.8rem' }}>
                            @{member.username}
                          </span>
                        </li>
                      ))}
                      {server.members.length > 50 && (
                        <li style={{ padding: '0.5rem', color: '#888' }}>
                          ...and {server.members.length - 50} more
                        </li>
                      )}
                    </ul>
                  )}
                </div>
              )}

              {activeTab === 'channels' && (
                <div style={{ maxHeight: '300px', overflow: 'auto' }}>
                  {server.channels.length === 0 ? (
                    <p>No channels found</p>
                  ) : (
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      {server.channels.map((channel) => (
                        <li
                          key={channel.id}
                          style={{
                            padding: '0.5rem',
                            borderBottom: '1px solid #333',
                          }}
                        >
                          <span style={{ color: '#888', marginRight: '0.5rem' }}>
                            #{channel.id}
                          </span>
                          {channel.name}
                          <span style={{ color: '#666', fontSize: '0.8rem', marginLeft: '0.5rem' }}>
                            ({channel.type})
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {activeTab === 'permissions' && (
                <div>
                  {permissionMessage && (
                    <div
                      style={{
                        padding: '0.5rem',
                        marginBottom: '1rem',
                        backgroundColor: permissionMessage.startsWith('Error') ? '#ff6b6b22' : '#22c55e22',
                        borderRadius: '4px',
                        color: permissionMessage.startsWith('Error') ? '#ff6b6b' : '#22c55e',
                      }}
                    >
                      {permissionMessage}
                    </div>
                  )}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '0.5rem' }}>
                    {Object.entries(server.permissions).map(([perm, value]) => (
                      <div
                        key={perm}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '0.5rem',
                          backgroundColor: '#252525',
                          borderRadius: '4px',
                        }}
                      >
                        <span style={{ textTransform: 'capitalize', fontSize: '0.9rem' }}>
                          {perm.replace(/_/g, ' ')}
                        </span>
                        <button
                          onClick={() => updatePermission(perm, !value)}
                          style={{
                            padding: '0.25rem 0.5rem',
                            fontSize: '0.8rem',
                            backgroundColor: value ? '#22c55e' : '#666',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#fff',
                            cursor: 'pointer',
                          }}
                        >
                          {value ? 'Revoke' : 'Grant'}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}