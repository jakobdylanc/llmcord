import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import yaml from 'js-yaml'
import { toast } from 'react-hot-toast'
import Modal from './Modal'
import { ArrowLeftIcon, TrashIcon } from '@heroicons/react/24/outline'

const API_BASE = '/api'

interface TaskDetail {
  name: string
  filename: string
  config: Record<string, any>
  status: string
}

interface TaskEditorProps {
  token: string
  taskName?: string | null  // null for new task
  onSave?: () => void
  onSaveWithName?: (name: string) => void  // Called after creating new task with the name
  onCancel?: () => void
}

// Default template for new tasks
const TEMPLATE_CONFIG = {
  name: 'new-task',
  enabled: true,
  cron: '0 * * * *',
  'user_id|channel_id': 1234567890,
  model: 'google/gemini-2.5-flash',
  prompt: 'Your task prompt here',
  tools: [],
  persona: ''
}

export default function TaskEditor({ token, taskName, onSave, onSaveWithName, onCancel }: TaskEditorProps) {
  const [name, setName] = useState(taskName || '')
  const [configText, setConfigText] = useState('')
  const [originalName, setOriginalName] = useState(taskName || '')
  const [loading, setLoading] = useState(!!taskName)
  const [saving, setSaving] = useState(false)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  
  // Flexbox auto-resize: editor fills available space automatically
  // No manual resize logic needed - browser handles it natively

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Load existing task if editing
  useEffect(() => {
    if (taskName) {
      fetchTask(taskName)
    } else if (taskName === undefined || taskName === null || taskName === '') {
      // New task - use template only when taskName is explicitly undefined/null/empty
      // This prevents re-loading template when switching from new to edit mode
      if (!originalName) {
        setConfigText(yaml.dump(TEMPLATE_CONFIG, { indent: 2, lineWidth: -1 }))
      }
    }
  }, [taskName])

  // Keep internal state in sync when prop changes (for navigation after save)
  useEffect(() => {
    if (taskName !== originalName) {
      setName(taskName || '')
      setOriginalName(taskName || '')
    }
  }, [taskName])

  const fetchTask = async (taskName: string) => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/tasks/${taskName}`, authHeaders)
      const task: TaskDetail = res.data
      setName(task.name)
      setOriginalName(task.name)
      // Convert config to YAML string
      setConfigText(yaml.dump(task.config, { indent: 2, lineWidth: -1 }))
    } catch (err: any) {
      console.error('Failed to fetch task:', err)
      toast.error(err.response?.data?.detail || 'Failed to load task')
    } finally {
      setLoading(false)
    }
  }

  const validateYaml = (text: string): { valid: boolean; error?: string; parsed?: any } => {
    // Step 1: Check for common lint issues before parsing
    const lines = text.split('\n')
    
    // Check for tabs (YAML requires spaces for indentation)
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('\t')) {
        return { valid: false, error: `Line ${i + 1}: Use spaces for indentation, not tabs` }
      }
    }
    
    // Check for inconsistent indentation (must be multiples of 2)
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      if (line.trim() === '' || line.trim().startsWith('#')) continue
      const indent = line.search(/\S/)
      if (indent > 0 && indent % 2 !== 0) {
        return { valid: false, error: `Line ${i + 1}: Indentation must be a multiple of 2 spaces` }
      }
    }
    
    // Step 2: Parse YAML
    try {
      const parsed: any = yaml.load(text)
      if (!parsed || typeof parsed !== 'object') {
        return { valid: false, error: 'Task config must be a YAML object' }
      }
      
      // FIX: JavaScript loses precision for large integers (> 2^53)
      // Discord IDs have 17-19 digits, but JS Number only has 52-bit mantissa (~16 digits)
      // Example: 1478227992747704442 → 1478227992747704300
      // Solution: Preserve ID fields as strings since Discord IDs are numeric strings anyway
      const idFields = ['user_id', 'channel_id', 'user_id|channel_id']
      for (const field of idFields) {
        if (field in parsed && parsed[field] !== undefined && parsed[field] !== null) {
          parsed[field] = String(parsed[field])
        }
      }
      
      // Step 3: Check for duplicate keys (yaml.load doesn't detect this)
      const seenKeys: Record<string, number> = {}
      for (const line of lines) {
        const match = line.match(/^(\s*)([^:#\s]+):/)
        if (match) {
          const key = match[2]
          if (seenKeys[key] !== undefined) {
            return { valid: false, error: `Duplicate key "${key}" - each key must be unique` }
          }
          seenKeys[key] = 1
        }
      }
      
      // Step 4: Validate required fields
      const requiredFields = ['name', 'enabled', 'cron', 'prompt']
      const missingFields = requiredFields.filter(field => !(field in parsed))
      if (missingFields.length > 0) {
        return { valid: false, error: `Missing required fields: ${missingFields.join(', ')}` }
      }
      
      // Check for at least one of the ID fields
      const hasUserId = 'user_id' in parsed
      const hasChannelId = 'channel_id' in parsed
      const hasCombinedId = 'user_id|channel_id' in parsed
      if (!hasUserId && !hasChannelId && !hasCombinedId) {
        return { valid: false, error: 'Missing required field: user_id, channel_id, or user_id|channel_id' }
      }
      
      return { valid: true, parsed }
    } catch (e: any) {
      // Extract line number from YAML error if available
      const message = e.message || 'Invalid YAML'
      const lineMatch = message.match(/line (\d+)/i)
      const lineInfo = lineMatch ? ` at line ${lineMatch[1]}` : ''
      return { valid: false, error: message + lineInfo }
    }
  }

  const handleSave = async () => {
    // Validate YAML first
    const validation = validateYaml(configText)
    if (!validation.valid) {
      setValidationError(validation.error || 'Invalid YAML')
      toast.error('YAML validation failed: ' + validation.error)
      return
    }
    setValidationError(null)

    // Check if name changed
    const config = validation.parsed
    if (name !== originalName) {
      config.name = name
    }

    try {
      setSaving(true)
      
      if (originalName && originalName !== name) {
        // Name changed - need to delete old and create new
        await axios.delete(`${API_BASE}/tasks/${originalName}`, authHeaders)
      }

      const isNewTask = !originalName || (originalName !== name)
      
      if (originalName && originalName === name) {
        // Update existing - wrap in { config: ... } for Pydantic model
        await axios.put(`${API_BASE}/tasks/${name}`, { config }, authHeaders)
        toast.success('Task saved!')
      } else {
        // Create new - wrap in { name, config } for Pydantic model
        await axios.post(`${API_BASE}/tasks`, { name, config }, authHeaders)
        toast.success('Task created!')
        onSaveWithName?.(name)
      }
      
      // Reload single task in bot's scheduler (not full reload - avoids disruption)
      // Skip for new tasks - they don't exist in scheduler yet
      if (!isNewTask) {
        try {
          await axios.post(`${API_BASE}/tasks/${name}/reload`, {}, authHeaders)
        } catch (reloadErr) {
          console.error('Failed to reload task:', reloadErr)
          // Don't fail the save if reload fails
        }
      }
      
      // Stay in editor after save
      if (originalName && originalName === name) {
        // Refresh the task data from server
        fetchTask(name)
      }
    } catch (err: any) {
      console.error('Failed to save task:', err)
      toast.error(err.response?.data?.detail || 'Failed to save task')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!originalName) return

    try {
      setSaving(true)
      await axios.delete(`${API_BASE}/tasks/${originalName}`, authHeaders)
      toast.success('Task deleted!')
      onSave?.()
    } catch (err: any) {
      console.error('Failed to delete task:', err)
      toast.error(err.response?.data?.detail || 'Failed to delete task')
    } finally {
      setSaving(false)
      setShowDeleteModal(false)
    }
  }

  const handleNameChange = (newName: string) => {
    // Sanitize name - only allow alphanumeric and hyphens
    const sanitized = newName.toLowerCase().replace(/[^a-z0-9-]/g, '-')
    setName(sanitized)
  }

  if (loading) {
    return <div style={{ padding: '2rem' }}>Loading task...</div>
  }

  return (
    <div ref={containerRef} style={{ padding: '1rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button 
            onClick={onCancel}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem' }}
          >
            <ArrowLeftIcon style={{ width: '1.25rem', height: '1.25rem' }} />
          </button>
          <h2 style={{ margin: 0 }}>
            {originalName ? `Edit: ${originalName}` : 'New Task'}
          </h2>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {originalName && (
            <button 
              onClick={() => setShowDeleteModal(true)}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: '#dc2626' }}
            >
              <TrashIcon style={{ width: '1rem', height: '1rem' }} />
              Delete
            </button>
          )}
          <button 
            onClick={handleSave}
            disabled={saving || !name}
            style={{ backgroundColor: '#22c55e', opacity: saving || !name ? 0.5 : 1 }}
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      {/* Name input */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
          Task Name
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => handleNameChange(e.target.value)}
          placeholder="task-name"
          style={{ width: '100%', maxWidth: '400px' }}
        />
        <p style={{ fontSize: '0.8rem', color: '#888', marginTop: '0.25rem' }}>
          Only lowercase letters, numbers, and hyphens allowed
        </p>
      </div>

      {/* YAML Editor */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
          YAML Configuration
          {validationError && (
            <span style={{ color: '#ff6b6b', fontWeight: 'normal', marginLeft: '0.5rem' }}>
              - {validationError}
            </span>
          )}
        </label>
        {/* Editor and help side by side */}
        <div style={{ flex: 1, display: 'flex', gap: '1rem', minHeight: 0 }}>
          {/* Editor container - flexbox auto-resize */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <textarea
              value={configText}
              onChange={(e) => {
                setConfigText(e.target.value)
                setValidationError(null)
              }}
              placeholder="name: my-task&#10;enabled: true&#10;cron: '0 * * * *'&#10;..."
              style={{
                flex: 1,
                minHeight: '200px',
                fontFamily: 'monospace',
                fontSize: '0.9rem',
                padding: '1rem',
                backgroundColor: '#1a1a1a',
                color: '#ddd',
                border: `1px solid ${validationError ? '#ff6b6b' : '#444'}`,
                borderRadius: '8px',
                resize: 'none'
              }}
            />
          </div>
          {/* Help text - right side */}
          <div style={{ width: '280px', padding: '1rem', backgroundColor: '#252525', borderRadius: '8px', fontSize: '0.85rem', flexShrink: 0, overflow: 'auto' }}>
            <strong>Task Fields:</strong>
            <ul style={{ margin: '0.5rem 0 0 0', paddingLeft: '1.5rem' }}>
              <li><code>name</code> - Task identifier</li>
              <li><code>enabled</code> - true/false to enable/disable</li>
              <li><code>cron</code> - Cron schedule (e.g., "0 * * * *" = every hour)</li>
              <li><code>user_id|channel_id</code> - Discord user or channel ID</li>
              <li><code>model</code> - LLM model to use</li>
              <li><code>prompt</code> - Task prompt</li>
              <li><code>tools</code> - List of tools to enable</li>
              <li><code>persona</code> - Persona name to use</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onConfirm={handleDelete}
        title="Delete Task"
        message={`Are you sure you want to delete "${originalName}"? This action cannot be undone.`}
        variant="danger"
        confirmText="Delete"
      />
    </div>
  )
}