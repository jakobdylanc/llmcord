import { useState, useEffect } from 'react'
import { 
  HomeIcon, 
  Cog6ToothIcon, 
  ServerStackIcon, 
  UserIcon,
  ListBulletIcon,
  CodeBracketIcon
} from '@heroicons/react/24/outline'

type TabType = 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
  onLogout: () => void
}

interface NavItem {
  id: TabType
  label: string
  icon: React.ComponentType<{ style?: React.CSSProperties }>
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: HomeIcon },
  { id: 'config', label: 'Config', icon: Cog6ToothIcon },
  { id: 'servers', label: 'Servers', icon: ServerStackIcon },
  { id: 'personas', label: 'Personas', icon: UserIcon },
  { id: 'tasks', label: 'Tasks', icon: ListBulletIcon },
  { id: 'skills', label: 'Skills', icon: CodeBracketIcon },
]

export default function Sidebar({ activeTab, onTabChange, onLogout }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Responsive: collapse to icons on narrow screens
  useEffect(() => {
    const handleResize = () => {
      setIsCollapsed(window.innerWidth <= 768)
    }

    handleResize() // Check on mount
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <aside
      style={{
        width: isCollapsed ? '60px' : '200px',
        minWidth: isCollapsed ? '60px' : '200px',
        backgroundColor: '#1a1a1a',
        padding: '1rem 0.5rem',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        boxSizing: 'border-box',
      }}
    >
      {/* Logo/Title */}
      <div
        style={{
          padding: '0.5rem 1rem',
          marginBottom: '1rem',
          borderBottom: '1px solid #333',
        }}
      >
        {!isCollapsed && (
          <h2 style={{ margin: 0, fontSize: '1.2rem', color: '#fff' }}>
            GPT Discord Bot
          </h2>
        )}
        {isCollapsed && (
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <Cog6ToothIcon style={{ width: '1.5rem', height: '1.5rem', color: '#646cff' }} />
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav style={{ flex: 1 }}>
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onTabChange(item.id)}
            title={isCollapsed ? item.label : undefined}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              gap: isCollapsed ? 0 : '0.75rem',
              justifyContent: isCollapsed ? 'center' : 'flex-start',
              padding: isCollapsed ? '0.75rem' : '0.75rem 1rem',
              marginBottom: '0.25rem',
              backgroundColor: activeTab === item.id ? '#646cff' : 'transparent',
              color: activeTab === item.id ? '#fff' : '#aaa',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'background-color 0.2s',
              fontSize: '1rem',
            }}
            onMouseEnter={(e) => {
              if (activeTab !== item.id) {
                e.currentTarget.style.backgroundColor = '#2a2a2a'
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== item.id) {
                e.currentTarget.style.backgroundColor = 'transparent'
              }
            }}
          >
            <item.icon style={{ width: '1.25rem', height: '1.25rem' }} />
            {!isCollapsed && <span>{item.label}</span>}
          </button>
        ))}
      </nav>

      {/* Logout Button */}
      <div style={{ borderTop: '1px solid #333', paddingTop: '1rem' }}>
        <button
          onClick={onLogout}
          title={isCollapsed ? 'Logout' : undefined}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: isCollapsed ? 0 : '0.75rem',
            justifyContent: isCollapsed ? 'center' : 'flex-start',
            padding: isCollapsed ? '0.75rem' : '0.75rem 1rem',
            backgroundColor: 'transparent',
            color: '#ff6b6b',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '1rem',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#2a2a2a'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent'
          }}
        >
          <span style={{ fontSize: '1.2rem' }}>🚪</span>
          {!isCollapsed && <span>Logout</span>}
        </button>
      </div>
    </aside>
  )
}