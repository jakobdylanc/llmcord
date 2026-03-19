import { useState, useEffect, createContext, useContext } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useSearchParams } from 'react-router-dom'
import { Toaster, toast } from 'react-hot-toast'
import Login from './components/Login'
import Dashboard from './components/Dashboard'
import ServerDrawer from './components/ServerDrawer'

// Export toast for use in other components
export { toast }

// Auth context for storing JWT token
interface AuthContextType {
  token: string | null
  setToken: (token: string | null) => void
}

const AuthContext = createContext<AuthContextType>({ token: null, setToken: () => {} })

export const useAuth = () => useContext(AuthContext)

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { token } = useAuth()
  if (!token) {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

// Component to handle the ServerDrawer with URL sync
function AppContent() {
  const { token } = useAuth()
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedServerId = searchParams.get('server')

  const openServerDrawer = (serverId: string) => {
    setSearchParams({ server: serverId })
  }

  const closeServerDrawer = () => {
    setSearchParams({})
  }

  return (
    <>
      <Dashboard onOpenServer={openServerDrawer} onCloseServer={closeServerDrawer} />
      {selectedServerId && token && (
        <ServerDrawer
          serverId={selectedServerId}
          token={token}
          onClose={closeServerDrawer}
        />
      )}
    </>
  )
}

function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))

  useEffect(() => {
    if (token) {
      localStorage.setItem('token', token)
    } else {
      localStorage.removeItem('token')
    }
  }, [token])

  return (
    <AuthContext.Provider value={{ token, setToken }}>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#333',
            color: '#fff',
          },
          success: {
            style: {
              background: '#10b981',
            },
          },
          error: {
            style: {
              background: '#ef4444',
            },
          },
        }}
      />
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <AppContent />
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </AuthContext.Provider>
  )
}

export default App