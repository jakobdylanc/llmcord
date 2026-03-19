import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth, toast } from '../App'
import axios from 'axios'

const API_BASE = '/api'

export default function Login() {
  const { setToken } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [isSetup, setIsSetup] = useState(false)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  // Check if setup is needed on mount
  useEffect(() => {
    axios.get(`${API_BASE}/auth/has-users`)
      .then(res => {
        setIsSetup(!res.data.has_users)
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    
    try {
      const endpoint = isSetup ? '/auth/setup' : '/auth/login'
      const res = await axios.post(`${API_BASE}${endpoint}`, {
        username,
        password,
      })
      
      setToken(res.data.access_token)
      toast.success(isSetup ? 'Admin account created!' : 'Login successful!')
      navigate('/')
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Authentication failed')
      setSubmitting(false)
    }
  }

  if (loading) {
    return <div>Loading...</div>
  }

  return (
    <div className="card">
      <h1>{isSetup ? 'Setup Admin Account' : 'Login'}</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div>
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" disabled={submitting}>
          {submitting ? 'Please wait...' : (isSetup ? 'Create Admin' : 'Login')}
        </button>
      </form>
    </div>
  )
}