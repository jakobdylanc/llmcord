import { useState, useEffect } from 'react'
import axios from 'axios'
import { ArrowPathIcon, CodeBracketIcon, InformationCircleIcon } from '@heroicons/react/24/outline'

const API_BASE = '/api'

interface SkillParameter {
  name: string
  type: string
  required: boolean
  description: string
}

interface SkillInfo {
  name: string
  filename: string
  description: string | null
  parameters: SkillParameter[]
}

interface SkillsListProps {
  token: string
  onSelectSkill?: (name: string) => void
}

export default function SkillsList({ token, onSelectSkill }: SkillsListProps) {
  const [skills, setSkills] = useState<SkillInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchSkills()
  }, [])

  const fetchSkills = async () => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/skills`, authHeaders)
      setSkills(res.data || [])
      setError(null)
    } catch (err: any) {
      console.error('Failed to fetch skills:', err)
      setError(err.response?.data?.detail || 'Failed to load skills')
    } finally {
      setLoading(false)
    }
  }

  const handleSkillClick = (name: string) => {
    const newSelection = selectedSkill === name ? null : name
    setSelectedSkill(newSelection)
    if (onSelectSkill) {
      onSelectSkill(newSelection || '')
    }
  }

  if (loading) {
    return <div>Loading skills...</div>
  }

  if (error) {
    return (
      <div style={{ padding: '1rem' }}>
        <p style={{ color: '#ff6b6b' }}>{error}</p>
        <button onClick={fetchSkills}>Retry</button>
      </div>
    )
  }

  return (
    <div style={{ padding: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <CodeBracketIcon style={{ width: '1.5rem', height: '1.5rem' }} />
          Skills (Tools)
        </h2>
        <button 
          onClick={fetchSkills} 
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <ArrowPathIcon style={{ width: '1rem', height: '1rem' }} />
          Refresh
        </button>
      </div>

      <p style={{ color: '#888', marginBottom: '1rem', fontSize: '0.875rem' }}>
        Read-only view of available skills. Modification is not in scope (future consideration).
      </p>

      {skills.length === 0 ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#888' }}>
          <p>No skills found.</p>
          <p>Skill files should be in <code>bot/llm/tools/skills/</code></p>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '1rem' }}>
          {skills.map((skill) => (
            <div
              key={skill.name}
              style={{
                border: '1px solid #444',
                borderRadius: '8px',
                padding: '1rem',
                backgroundColor: selectedSkill === skill.name ? '#2a2a2a' : '#1e1e1e',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
              }}
              onClick={() => handleSkillClick(skill.name)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.125rem', fontWeight: '600' }}>
                    {skill.name}
                  </h3>
                  {skill.description && (
                    <p style={{ margin: 0, color: '#aaa', fontSize: '0.875rem' }}>
                      {skill.description}
                    </p>
                  )}
                </div>
                <InformationCircleIcon style={{ width: '1.25rem', height: '1.25rem', color: '#666' }} />
              </div>

              {selectedSkill === skill.name && skill.parameters.length > 0 && (
                <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #444' }}>
                  <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.875rem', color: '#888' }}>
                    Parameters
                  </h4>
                  <table style={{ width: '100%', fontSize: '0.875rem', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ textAlign: 'left', color: '#888' }}>
                        <th style={{ padding: '0.25rem 0.5rem 0.5rem 0' }}>Name</th>
                        <th style={{ padding: '0.25rem 0.5rem 0.5rem 0' }}>Type</th>
                        <th style={{ padding: '0.25rem 0.5rem 0.5rem 0' }}>Required</th>
                        <th style={{ padding: '0.25rem 0 0.5rem 0' }}>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {skill.parameters.map((param, idx) => (
                        <tr key={idx}>
                          <td style={{ padding: '0.25rem 0.5rem 0.25rem 0' }}>
                            <code style={{ backgroundColor: '#333', padding: '0.125rem 0.375rem', borderRadius: '4px' }}>
                              {param.name}
                            </code>
                          </td>
                          <td style={{ padding: '0.25rem 0.5rem 0.25rem 0', color: '#7dd3fc' }}>
                            {param.type}
                          </td>
                          <td style={{ padding: '0.25rem 0.5rem 0.25rem 0' }}>
                            {param.required ? (
                              <span style={{ color: '#f87171', fontWeight: '500' }}>Yes</span>
                            ) : (
                              <span style={{ color: '#888' }}>No</span>
                            )}
                          </td>
                          <td style={{ padding: '0.25rem 0 0.25rem 0', color: '#aaa' }}>
                            {param.description}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}