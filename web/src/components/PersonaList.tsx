import { useState, useEffect } from 'react'
import axios from 'axios'
import { ArrowPathIcon } from '@heroicons/react/24/outline'

const API_BASE = '/api'

interface PersonaInfo {
  name: string
  filename: string
  description: string | null
}

interface PersonaListProps {
  token: string
  onSelectPersona?: (name: string) => void
  onCreateNew?: () => void
}

export default function PersonaList({ token, onSelectPersona, onCreateNew }: PersonaListProps) {
  const [personas, setPersonas] = useState<PersonaInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchPersonas()
  }, [])

  const fetchPersonas = async () => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/personas`, authHeaders)
      setPersonas(res.data || [])
      setError(null)
    } catch (err: any) {
      console.error('Failed to fetch personas:', err)
      setError(err.response?.data?.detail || 'Failed to load personas')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div>Loading personas...</div>
  }

  if (error) {
    return (
      <div style={{ padding: '1rem' }}>
        <p style={{ color: '#ff6b6b' }}>{error}</p>
        <button onClick={fetchPersonas}>Retry</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2>Personas</h2>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {onCreateNew && (
            <button onClick={onCreateNew} style={{ backgroundColor: '#22c55e' }}>
              ➕ Add New
            </button>
          )}
          <button onClick={fetchPersonas} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ArrowPathIcon style={{ width: '1rem', height: '1rem' }} />
            Refresh
          </button>
        </div>
      </div>

      {personas.length === 0 ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#888' }}>
          <p>No personas found.</p>
          <p>Create a persona file in <code>bot/config/personas/</code></p>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '1rem' }}>
          {personas.map((persona) => (
            <div
              key={persona.name}
              onClick={() => onSelectPersona?.(persona.name)}
              style={{
                backgroundColor: '#1a1a1a',
                borderRadius: '8px',
                padding: '1rem',
                cursor: onSelectPersona ? 'pointer' : 'default',
                transition: 'background-color 0.2s',
                border: '1px solid #333',
              }}
              onMouseEnter={(e) => {
                if (onSelectPersona) {
                  e.currentTarget.style.backgroundColor = '#252525'
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#1a1a1a'
              }}
            >
              <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem' }}>
                {persona.name}
              </h3>
              {persona.description && (
                <p style={{ margin: 0, color: '#888', fontSize: '0.9rem' }}>
                  {persona.description}
                </p>
              )}
              <p style={{ margin: '0.5rem 0 0 0', color: '#666', fontSize: '0.8rem' }}>
                {persona.filename}
              </p>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: '#252525', borderRadius: '8px' }}>
        <h4 style={{ margin: '0 0 0.5rem 0' }}>Add New Persona</h4>
        <p style={{ margin: 0, color: '#888', fontSize: '0.9rem' }}>
          Create a new persona file in <code>bot/config/personas/</code> with a <code>.md</code> extension.
        </p>
      </div>
    </div>
  )
}