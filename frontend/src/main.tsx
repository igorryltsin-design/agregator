import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import 'bootstrap/dist/css/bootstrap.min.css'
import './theme.css'
import App from './pages/App'
import Catalogue from './pages/Catalogue'
import GraphPage from './pages/GraphPage'
import StatsPage from './pages/StatsPage'
import SettingsPage from './pages/SettingsPage'
import IngestPage from './pages/IngestPage'
import DocumentChatPage from './pages/DocumentChatPage'
import OsintPage from './pages/OsintPage'
import LoginPage from './pages/LoginPage'
import ProfilePage from './pages/ProfilePage'
import UsersPage from './pages/UsersPage'
import AdminTasksPage from './pages/AdminTasksPage'
import AdminLogsPage from './pages/AdminLogsPage'
import AdminCollectionsPage from './pages/AdminCollectionsPage'
import AdminAiMetricsPage from './pages/AdminAiMetricsPage'
import AdminServiceStatusPage from './pages/AdminServiceStatusPage'
import ToastProvider from './ui/Toasts'
import AuthProvider from './ui/Auth'
import { AppErrorBoundary, PageErrorBoundary } from './ui/ErrorBoundary'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <PageErrorBoundary label="Каталог"><Catalogue /></PageErrorBoundary> },
      { path: 'doc-chat', element: <PageErrorBoundary label="Чат по документу"><DocumentChatPage /></PageErrorBoundary> },
      { path: 'osint', element: <PageErrorBoundary label="OSINT"><OsintPage /></PageErrorBoundary> },
      { path: 'graph', element: <PageErrorBoundary label="Граф"><GraphPage /></PageErrorBoundary> },
      { path: 'stats', element: <PageErrorBoundary label="Статистика"><StatsPage /></PageErrorBoundary> },
      { path: 'settings', element: <PageErrorBoundary label="Настройки"><SettingsPage /></PageErrorBoundary> },
      { path: 'ingest', element: <PageErrorBoundary label="Импорт"><IngestPage /></PageErrorBoundary> },
      { path: 'profile', element: <PageErrorBoundary label="Профиль"><ProfilePage /></PageErrorBoundary> },
      { path: 'users', element: <PageErrorBoundary label="Пользователи"><UsersPage /></PageErrorBoundary> },
      { path: 'admin/tasks', element: <PageErrorBoundary label="Админ: задачи"><AdminTasksPage /></PageErrorBoundary> },
      { path: 'admin/logs', element: <PageErrorBoundary label="Админ: логи"><AdminLogsPage /></PageErrorBoundary> },
      { path: 'admin/status', element: <PageErrorBoundary label="Админ: состояние"><AdminServiceStatusPage /></PageErrorBoundary> },
      { path: 'admin/collections', element: <PageErrorBoundary label="Админ: коллекции"><AdminCollectionsPage /></PageErrorBoundary> },
      { path: 'admin/ai-metrics', element: <PageErrorBoundary label="Админ: AI метрики"><AdminAiMetricsPage /></PageErrorBoundary> },
    ],
  },
  { path: '/login', element: <LoginPage /> },
], { basename: '/app' })

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <ToastProvider>
        <AuthProvider>
          <RouterProvider router={router} />
        </AuthProvider>
      </ToastProvider>
    </AppErrorBoundary>
  </React.StrictMode>
)
