import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = '/api'

interface ConfigField {
  key: string
  value: any
  editable: boolean
  type: string
}

interface ConfigEditorProps {
  token: string
}

export default function ConfigEditor({ token }: ConfigEditorProps) {
  const [config, setConfig] = useState<ConfigField[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchConfig()
  }, [])

  const fetchConfig = async () => {
    try {
      const res = await axios.get(`${API_BASE}/config`, authHeaders)
      const apiData = res.data
      
      // Transform API response to frontend format
      const fields: ConfigField[] = []
      
      // Add editable fields
      for (const key of apiData.editable_fields || []) {
        const value = getNestedValue(apiData.config, key)
        fields.push({ key, value, editable: true, type: typeof value })
      }
      
      // Add read-only fields (collapsed)
      for (const key of apiData.read_only_fields || []) {
        const value = getNestedValue(apiData.config, key)
        fields.push({ key, value, editable: false, type: typeof value })
      }
      
      setConfig(fields)
    } catch (err) {
      console.error('Failed to fetch config:', err)
    } finally {
      setLoading(false)
    }
  }

  // Helper to get nested value from config object
  const getNestedValue = (obj: any, path: string): any => {
    if (!obj) return undefined
    return path.split('.').reduce((acc, part) => acc && acc[part], obj)
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage('')
    
    try {
      const updates = config
        .filter(f => f.editable)
        .map(f => ({ key: f.key, value: f.value }))
      
      await axios.put(`${API_BASE}/config`, { fields: updates }, authHeaders)
      setMessage('Configuration saved successfully!')
    } catch (err: any) {
      setMessage(err.response?.data?.detail || 'Failed to save config')
    } finally {
      setSaving(false)
    }
  }

  // Apply: reload config in-memory (no file write)
  const handleApply = async () => {
    setSaving(true)
    setMessage('')
    
    try {
      await axios.post(`${API_BASE}/refresh`, {}, authHeaders)
      setMessage('Configuration applied (in-memory reload)!')
      fetchConfig()
    } catch (err: any) {
      setMessage(err.response?.data?.detail || 'Failed to apply config')
    } finally {
      setSaving(false)
    }
  }

  // Save&Apply: writes to file AND reloads in-memory
  const handleSaveAndApply = async () => {
    setSaving(true)
    setMessage('')
    
    try {
      // First save to file
      const updates = config
        .filter(f => f.editable)
        .map(f => ({ key: f.key, value: f.value }))
      
      await axios.put(`${API_BASE}/config`, { fields: updates }, authHeaders)
      
      // Then reload in-memory
      await axios.post(`${API_BASE}/refresh`, {}, authHeaders)
      
      setMessage('Configuration saved and applied!')
      fetchConfig()
    } catch (err: any) {
      setMessage(err.response?.data?.detail || 'Failed to save and apply config')
    } finally {
      setSaving(false)
    }
  }

  const updateField = (key: string, value: any) => {
    setConfig(prev => prev.map(f => 
      f.key === key ? { ...f, value } : f
    ))
  }

  const renderInput = (field: ConfigField) => {
    if (!field.editable) {
      return <span>{String(field.value)}</span>
    }

    switch (field.type) {
      case 'boolean':
        return (
          <input
            type="checkbox"
            checked={Boolean(field.value)}
            onChange={(e) => updateField(field.key, e.target.checked)}
          />
        )
      case 'number':
        return (
          <input
            type="number"
            value={field.value}
            onChange={(e) => updateField(field.key, Number(e.target.value))}
          />
        )
      default:
        return (
          <input
            type="text"
            value={field.value}
            onChange={(e) => updateField(field.key, e.target.value)}
          />
        )
    }
  }

  if (loading) {
    return <div>Loading...</div>
  }

  return (
    <div>
      <h2>Configuration</h2>
      
      {message && (
        <p style={{ 
          color: message.includes('success') ? '#10ac84' : '#ee5253',
          marginBottom: '1rem'
        }}>
          {message}
        </p>
      )}

      <div style={{ 
        backgroundColor: '#1a1a1a', 
        borderRadius: '8px', 
        padding: '1rem',
        marginBottom: '1rem'
      }}>
        {config.map((field) => (
          <div key={field.key} style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            padding: '0.75rem 0',
            borderBottom: '1px solid #333'
          }}>
            <div>
              <strong>{field.key}</strong>
              {!field.editable && (
                <span style={{ 
                  fontSize: '0.8em', 
                  color: '#888', 
                  marginLeft: '0.5rem' 
                }}>
                  (read-only)
                </span>
              )}
            </div>
            <div style={{ minWidth: '200px', textAlign: 'right' }}>
              {renderInput(field)}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <button onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button onClick={handleApply} disabled={saving}>
          {saving ? 'Applying...' : 'Apply'}
        </button>
        <button onClick={handleSaveAndApply} disabled={saving}>
          {saving ? 'Saving...' : 'Save&Apply'}
        </button>
      </div>
    </div>
  )
}