import { useEffect, useRef } from 'react'

interface ModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  variant?: 'danger' | 'warning' | 'info'
}

export default function Modal({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'danger'
}: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isOpen, onClose])

  // Close on click outside
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === modalRef.current) {
      onClose()
    }
  }

  if (!isOpen) return null

  const variantStyles = {
    danger: {
      confirmBg: '#ef4444',
      confirmHover: '#dc2626',
      icon: '⚠️'
    },
    warning: {
      confirmBg: '#f59e0b',
      confirmHover: '#d97706',
      icon: '⚡'
    },
    info: {
      confirmBg: '#3b82f6',
      confirmHover: '#2563eb',
      icon: 'ℹ️'
    }
  }

  const styles = variantStyles[variant]

  return (
    <div
      ref={modalRef}
      onClick={handleBackdropClick}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
      }}
    >
      <div
        style={{
          backgroundColor: '#1a1a1a',
          borderRadius: '8px',
          padding: '1.5rem',
          maxWidth: '400px',
          width: '90%',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
          border: '1px solid #333',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <span style={{ fontSize: '1.5rem' }}>{styles.icon}</span>
          <h3 style={{ margin: 0, color: '#fff', fontSize: '1.125rem' }}>{title}</h3>
        </div>

        {/* Message */}
        <p style={{ color: '#a1a1aa', marginBottom: '1.5rem', lineHeight: 1.5 }}>
          {message}
        </p>

        {/* Actions */}
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'transparent',
              border: '1px solid #444',
              borderRadius: '6px',
              color: '#d4d4d8',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            {cancelText}
          </button>
          <button
            onClick={() => {
              onConfirm()
              onClose()
            }}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: styles.confirmBg,
              border: 'none',
              borderRadius: '6px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: 500,
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = styles.confirmHover}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = styles.confirmBg}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}