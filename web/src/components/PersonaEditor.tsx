import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'
import { toast } from 'react-hot-toast'
import Modal from './Modal'
import { ArrowLeftIcon, DocumentArrowDownIcon, TrashIcon } from '@heroicons/react/24/outline'

const API_BASE = '/api'

interface PersonaDetail {
  name: string
  filename: string
  content: string
  word_count: number
}

interface PersonaUsage {
  name: string
  filename: string
  type: 'task' | 'model'
  schedule?: string
}

interface PersonaEditorProps {
  token: string
  personaName?: string | null  // null for new persona
  onSave?: () => void
  onSaveWithName?: (name: string) => void  // Called after creating new persona with the name
  onCancel?: () => void
}

// Default template content from persona-example.md
const TEMPLATE_CONTENT = `**Role:** You are a {role_description}.

**Operational Rules (CRITICAL):**
1. **Time Lock:** Confirm today's date \`{date}\`.
   - If it's Monday, {rule_if_monday}.
   - If it's Tuesday‑Saturday, {rule_if_weekday}.
   - **Strictly disallow** referencing non‑trading days or future dates with fabricated data.

2. **Data Retrieval Strategy:**
   - **Step 1:** Call \`{tool_name1}\` to obtain the required data (must be the first call).
   - **Step 2 (optional):** If additional context is needed, use \`{tool_name2}\`.
   - **Prohibit** emitting data or warnings before \`{tool_name1}\` has run.

3. **Discord Format Guidelines:**
   - **No Markdown tables** (avoid \`|--|\`).
   - **Emphasis tags:** Important info must be **bolded**.

4. **Language:** Must use English only.

**Output Structure:**
### 📅 {report_title}: [YYYY‑MM‑DD] ({context_info})

**Summary:** [Two‑sentence overview of key points]

**Data Highlights:** [Relevant data or observations]

**Insights (Optional):**  
- Insight 1  
- Insight 2  
- Insight 3

**Error Handling:**
- If \`{tool}\` returns no data → mark with ⚠️ Unable to retrieve accurate data for [date]; do **not** fabricate numbers.

**Current Context:**
- Today is \`{date}\`, Location is \`{location}\`.
`

export default function PersonaEditor({ token, personaName, onSave, onSaveWithName, onCancel }: PersonaEditorProps) {
  const [name, setName] = useState(personaName || '')
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(!!personaName)
  const [saving, setSaving] = useState(false)
  const [showPreview, setShowPreview] = useState(true)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [usage, setUsage] = useState<PersonaUsage[]>([])

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Load existing persona if editing
  useEffect(() => {
    if (personaName) {
      fetchPersona(personaName)
    }
  }, [personaName])

  const fetchPersona = async (name: string) => {
    setLoading(true)
    try {
      const res = await axios.get<PersonaDetail>(`${API_BASE}/personas/${name}`, authHeaders)
      setName(res.data.name)
      setContent(res.data.content)
      
      // Fetch usage info (21.5.3)
      try {
        const usageRes = await axios.get<PersonaUsage[]>(`${API_BASE}/personas/${name}/usage`, authHeaders)
        setUsage(usageRes.data)
      } catch {
        // Usage endpoint may not exist yet
        setUsage([])
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to load persona')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error('Persona name is required')
      return
    }

    // Validate name format
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      toast.error('Name can only contain letters, numbers, dash, and underscore')
      return
    }

    setSaving(true)

    try {
      const isNew = !personaName
      if (isNew) {
        await axios.post(`${API_BASE}/personas`, { name, content }, authHeaders)
        toast.success(`Persona "${name}" created successfully`)
        // Set the name so we switch from "new" mode to "edit" mode
        // The parent will update editingPersona state via onSaveWithName
        if (onSaveWithName) {
          onSaveWithName(name)
        }
      } else {
        await axios.put(`${API_BASE}/personas/${name}`, { content }, authHeaders)
        toast.success(`Persona "${name}" saved successfully`)
      }

      // Note: Apply/Refresh not needed - bot reads persona from disk each time
      // So we stay in editor mode instead of navigating back
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save persona')
    } finally {
      setSaving(false)
    }
  }

  const handleDeleteClick = () => {
    if (!personaName) return
    setShowDeleteModal(true)
  }

  const handleDeleteConfirm = async () => {
    if (!personaName) return

    setSaving(true)

    try {
      await axios.delete(`${API_BASE}/personas/${personaName}`, authHeaders)
      toast.success(`Persona "${personaName}" deleted`)
      onSave?.()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to delete persona')
    } finally {
      setSaving(false)
    }
  }

  const loadTemplate = () => {
    setContent(TEMPLATE_CONTENT)
  }

  if (loading) {
    return <div style={{ padding: '2rem' }}>Loading persona...</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '1rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>{personaName ? `Edit: ${personaName}` : 'New Persona'}</h2>
        <button onClick={onCancel} style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ArrowLeftIcon style={{ width: '1rem', height: '1rem' }} />
          Back
        </button>
      </div>

      {/* Usage info (21.5.3) - enhanced to show tasks AND models */}
      {usage.length > 0 && (
        <div style={{ 
          padding: '0.75rem', 
          backgroundColor: '#422006', 
          border: '1px solid #f59e0b',
          borderRadius: '4px',
          fontSize: '0.875rem'
        }}>
          <strong style={{ color: '#f59e0b' }}>⚠️ Used by {usage.length} item(s):</strong>
          
          {/* Separate tasks and models */}
          {usage.filter(u => u.type === 'task').length > 0 && (
            <>
              <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>Tasks:</div>
              <ul style={{ margin: '0.25rem 0 0 1.5rem', padding: 0 }}>
                {usage.filter(u => u.type === 'task').map((u) => (
                  <li key={u.filename}>
                    <code>{u.name}</code>
                    {u.schedule && <span style={{ color: '#9ca3af', marginLeft: '0.5rem' }}>schedule: {u.schedule}</span>}
                  </li>
                ))}
              </ul>
            </>
          )}
          
          {usage.filter(u => u.type === 'model').length > 0 && (
            <>
              <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>Models (config.yaml):</div>
              <ul style={{ margin: '0.25rem 0 0 1.5rem', padding: 0 }}>
                {usage.filter(u => u.type === 'model').map((u) => (
                  <li key={u.filename}>
                    <code>{u.name}</code>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {/* Name input for new personas */}
      {!personaName && (
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>Persona Name:</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-persona"
            style={{
              width: '100%',
              padding: '0.5rem',
              backgroundColor: '#252525',
              border: '1px solid #444',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <button
          onClick={handleSave}
          disabled={saving}
          style={{ padding: '0.5rem 1rem', backgroundColor: '#22c55e', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          Save
        </button>
        {!personaName && (
          <button
            onClick={loadTemplate}
            disabled={saving}
            style={{ padding: '0.5rem 1rem', backgroundColor: '#6366f1', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <DocumentArrowDownIcon style={{ width: '1rem', height: '1rem' }} />
            Use Template
          </button>
        )}
        {personaName && !personaName.endsWith('-example') && (
          <button
            onClick={handleDeleteClick}
            disabled={saving}
            style={{ padding: '0.5rem 1rem', backgroundColor: '#ef4444', marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <TrashIcon style={{ width: '1rem', height: '1rem' }} />
            Delete
          </button>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && personaName && (
        <Modal
          isOpen={showDeleteModal}
          onClose={() => setShowDeleteModal(false)}
          onConfirm={handleDeleteConfirm}
          title="Delete Persona"
          message={`Are you sure you want to delete "${personaName}"? This action cannot be undone.`}
          confirmText="Delete"
          cancelText="Cancel"
          variant="danger"
        />
      )}

      {/* Editor + Preview */}
      <div style={{ display: 'flex', gap: '1rem', flex: 1, minHeight: 0 }}>
        {/* Editor */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
            <span>Markdown Editor</span>
            <button onClick={() => setShowPreview(!showPreview)} style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}>
              {showPreview ? 'Hide' : 'Show'} Preview
            </button>
          </div>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Write your persona content in Markdown..."
            style={{
              flex: 1,
              resize: 'none',
              padding: '1rem',
              backgroundColor: '#252525',
              border: '1px solid #444',
              borderRadius: '4px',
              color: '#fff',
              fontFamily: 'monospace',
              fontSize: '0.9rem',
              lineHeight: '1.5',
            }}
          />
        </div>

        {/* Preview */}
        {showPreview && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <span style={{ marginBottom: '0.5rem' }}>Preview</span>
            <div
              style={{
                flex: 1,
                overflow: 'auto',
                padding: '1rem',
                backgroundColor: '#1a1a1a',
                border: '1px solid #444',
                borderRadius: '4px',
                color: '#ddd',
                fontSize: '0.9rem',
                lineHeight: '1.6',
              }}
            >
              <ReactMarkdown>{content || '*No content*'}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Word count */}
      <div style={{ color: '#666', fontSize: '0.8rem' }}>
        {content.split(/\s+/).filter(Boolean).length} words
      </div>
    </div>
  )
}