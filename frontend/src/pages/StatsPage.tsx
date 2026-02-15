import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { materialTypeRu, tagKeyRu } from '../utils/locale'
import LoadingState from '../ui/LoadingState'
import { EmptyState } from '../ui/EmptyState'
import { Chart, BarController, BarElement, CategoryScale, LinearScale, LogarithmicScale, LineController, LineElement, PointElement, ArcElement, Tooltip, Legend, DoughnutController } from 'chart.js'
Chart.register(BarController, BarElement, CategoryScale, LinearScale, LogarithmicScale, LineController, LineElement, PointElement, ArcElement, DoughnutController, Tooltip, Legend)

const formatScore = (value?: number) => (typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '—')

type StatKV = [string, number]
type StatsResponse = {
  authors: StatKV[]
  authors_cloud: StatKV[]
  years: StatKV[]
  types: StatKV[]
  exts: StatKV[]
  sizes: StatKV[]
  months: StatKV[]
  top_keywords: StatKV[]
  tag_keys: StatKV[]
  tag_values_cloud: StatKV[]
  weekdays: StatKV[]
  hours: StatKV[]
  avg_size_type: StatKV[]
  meta_presence: StatKV[]
  meta_missing?: StatKV[]
  total_files: number
  total_size_bytes: number
  collections_counts?: StatKV[]
  collections_total_size?: [string, number][]
  largest_files?: [string, number][]
  recent_counts?: { '7d'?: number; '30d'?: number }
  tags_summary?: {
    avg_per_file?: number
    with_tags?: number
    without_tags?: number
    total_tags?: number
  }
  authority_docs?: { id: number; score: number; title: string; author?: string | null; year?: string | null; collection_id?: number | null; collection_name?: string | null }[]
  authority_authors?: { name: string; score: number; count?: number }[]
  authority_topics?: { key: string; label: string; score: number; count?: number }[]
  authority_collections?: { collection_id?: number | null; name: string; score: number; count?: number }[]
  feedback_positive?: { file_id: number; title: string; author?: string | null; year?: string | null; collection_id?: number | null; collection_name?: string | null; weight: number; positive: number; negative: number; clicks: number; updated_at?: string | null }[]
  feedback_negative?: { file_id: number; title: string; author?: string | null; year?: string | null; collection_id?: number | null; collection_name?: string | null; weight: number; positive: number; negative: number; clicks: number; updated_at?: string | null }[]
  feedback_summary?: { total_files?: number; positive?: number; negative?: number }
}

export default function StatsPage(){
  const nav = useNavigate()
  const [data, setData] = useState<StatsResponse|null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string| null>(null)
  const [rangeMonths, setRangeMonths] = useState<number>(24)
  const [logScale, setLogScale] = useState<boolean>(false)
  const [cumulativeMonths, setCumulativeMonths] = useState<boolean>(false)
  const refTypes = useRef<HTMLCanvasElement|null>(null)
  const refMonths = useRef<HTMLCanvasElement|null>(null)
  const refYears = useRef<HTMLCanvasElement|null>(null)
  const refSizes = useRef<HTMLCanvasElement|null>(null)
  const refExts = useRef<HTMLCanvasElement|null>(null)
  const refWeekdays = useRef<HTMLCanvasElement|null>(null)
  const refHours = useRef<HTMLCanvasElement|null>(null)
  const refMeta = useRef<HTMLCanvasElement|null>(null)
  const refAvgSize = useRef<HTMLCanvasElement|null>(null)
  const refCollections = useRef<HTMLCanvasElement|null>(null)
  const refCollSize = useRef<HTMLCanvasElement|null>(null)
  const refLargest = useRef<HTMLCanvasElement|null>(null)
  const refAuthorsBar = useRef<HTMLCanvasElement|null>(null)
  const chartTypes = useRef<Chart | null>(null)
  const chartMonths = useRef<Chart | null>(null)
  const chartYears = useRef<Chart | null>(null)
  const chartSizes = useRef<Chart | null>(null)
  const chartExts = useRef<Chart | null>(null)
  const chartWeekdays = useRef<Chart | null>(null)
  const chartHours = useRef<Chart | null>(null)
  const chartMeta = useRef<Chart | null>(null)
  const chartAvg = useRef<Chart | null>(null)
  const chartCollections = useRef<Chart | null>(null)
  const chartCollSize = useRef<Chart | null>(null)
  const chartLargest = useRef<Chart | null>(null)
  const refTagKeys = useRef<HTMLCanvasElement|null>(null)
  const refTagVals = useRef<HTMLCanvasElement|null>(null)
  const chartTagKeys = useRef<Chart | null>(null)
  const chartTagVals = useRef<Chart | null>(null)
  const [selectedKey, setSelectedKey] = useState<string>('')
  const [tagVals, setTagVals] = useState<StatKV[]>([])
  const chartAuthorsBar = useRef<Chart | null>(null)
  useEffect(() => {
    const ac = new AbortController()
    ;(async ()=>{
      setLoading(true); setError(null)
      try{
        const r=await fetch('/api/stats', { signal: ac.signal })
        if (!r.ok) throw new Error('Ошибка загрузки статистики')
        const j: StatsResponse = await r.json(); setData(j)
      } catch (e:any) {
        if (e?.name !== 'AbortError') setError(String(e?.message||e))
      } finally{ setLoading(false) }
    })()
    return () => ac.abort()
  }, [])

  // Палитра считывается из CSS‑переменных темы
  const palette = useMemo(() => {
    const css = getComputedStyle(document.documentElement)
    return {
      primary: css.getPropertyValue('--primary-2').trim() || '#1f6feb',
      green: css.getPropertyValue('--primary-2').trim() || '#2ea043',
      accent: css.getPropertyValue('--accent').trim() || '#58a6ff',
      muted: css.getPropertyValue('--muted').trim() || '#8b949e',
      border: css.getPropertyValue('--border').trim() || '#30363d',
      gray: '#6e7781',
      yellow: '#e0a800',
      purple: '#8957e5',
      red: '#d73a49',
    }
  }, [document.documentElement.getAttribute('data-theme')])

  const numberFormatter = useMemo(() => new Intl.NumberFormat('ru-RU'), [])

  const commonOptions: any = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins:{ legend:{ display:false }, tooltip: { enabled: true } },
    scales:{ y:{ beginAtZero:true, type: logScale? 'logarithmic' : 'linear', grid:{ color: palette.border } }, x:{ grid:{ color: palette.border } } }
  }), [palette, logScale])
  const hbarOptions: any = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins:{ legend:{ display:false }, tooltip: { enabled: true } },
    scales:{ x:{ beginAtZero:true, type: logScale? 'logarithmic' : 'linear', grid:{ color: palette.border } }, y:{ grid:{ color: palette.border } } }
  }), [palette, logScale])
  useEffect(() => {
    if (!data) return
    const types = data.types || []
    if (refTypes.current) {
      chartTypes.current?.destroy()
      chartTypes.current = new Chart(refTypes.current, {
        type: 'bar',
        data: { labels: types.map((x:any)=>materialTypeRu(x[0])), datasets:[{ label:'По типам', data: types.map((x:any)=>x[1]), backgroundColor: palette.primary }] },
        options: {
          ...commonOptions,
          onClick: (_evt: any, els: any[]) => { if(!els?.length) return; const i=els[0].index; const raw = types?.[i]?.[0]; if (raw) nav(`/?type=${encodeURIComponent(String(raw))}`) }
        }
      })
    }
    const monthsRaw = (data.months||[]).slice(rangeMonths>0? -rangeMonths: undefined)
    const months = cumulativeMonths ? (function(){
      let acc = 0
      return monthsRaw.map(([m,v]: any) => { acc += Number(v||0); return [m, acc] as any })
    })() : monthsRaw
    if (refMonths.current) {
      chartMonths.current?.destroy()
      chartMonths.current = new Chart(refMonths.current, {
        type:'line',
        data:{ labels: months.map((x:any)=>x[0]), datasets:[{ label:'По месяцам', data: months.map((x:any)=>x[1]), borderColor: palette.green, backgroundColor:'rgba(46,160,67,0.3)', tension: 0.2, fill: true }] },
        options:{ ...commonOptions }
      })
    }
    const years = data.years || []
    if (refYears.current) {
      chartYears.current?.destroy()
      chartYears.current = new Chart(refYears.current, {
        type:'bar',
        data:{ labels: years.map((x:any)=>x[0]), datasets:[{ label:'По годам', data: years.map((x:any)=>x[1]), backgroundColor: palette.gray }] },
        options:{ ...commonOptions }
      })
    }
    const sizes = data.sizes || []
    if (refSizes.current) {
      chartSizes.current?.destroy()
      chartSizes.current = new Chart(refSizes.current, {
        type:'doughnut',
        data:{ labels: sizes.map((x:any)=>x[0]), datasets:[{ data: sizes.map((x:any)=>x[1]), backgroundColor:[palette.green,palette.primary,palette.yellow,palette.purple,palette.red,palette.gray] }] },
        options:{
          cutout: '60%',
          plugins:{
            legend:{position:'bottom'},
            tooltip: {
              callbacks: {
                label: (ctx: any) => {
                  const total = (ctx.dataset.data||[]).reduce((a:number,b:number)=>a+b,0) || 1
                  const v = ctx.raw as number; const p = Math.round((v*100)/total)
                  return `${ctx.label}: ${v} (${p}%)`
                }
              }
            }
          },
          maintainAspectRatio:false
        }
      })
    }
    const exts = data.exts || []
    if (refExts.current) {
      chartExts.current?.destroy()
      chartExts.current = new Chart(refExts.current, {
        type:'bar',
        data:{ labels: exts.map((x:any)=>x[0]), datasets:[{ label:'По расширениям', data: exts.map((x:any)=>x[1]), backgroundColor: palette.accent }] },
        options:{ ...commonOptions }
      })
    }
    const weekdays = data.weekdays || []
    if (refWeekdays.current) {
      chartWeekdays.current?.destroy()
      chartWeekdays.current = new Chart(refWeekdays.current, {
        type:'bar',
        data:{ labels: weekdays.map((x:any)=>x[0]), datasets:[{ label:'Дни недели', data: weekdays.map((x:any)=>x[1]), backgroundColor: palette.purple }] },
        options:{ ...commonOptions,
          onClick: (_evt:any, els:any[]) => { if(!els?.length) return; const i=els[0].index; const label=(chartWeekdays.current as any)?.data?.labels?.[i]; if(label) nav(`/?q=${encodeURIComponent('weekday:'+label)}`) }
        }
      })
    }
    const hours = data.hours || []
    if (refHours.current) {
      chartHours.current?.destroy()
      chartHours.current = new Chart(refHours.current, {
        type:'bar',
        data:{ labels: hours.map((x:any)=>x[0]), datasets:[{ label:'Часы', data: hours.map((x:any)=>x[1]), backgroundColor: palette.yellow }] },
        options:{ ...commonOptions,
          onClick: (_evt:any, els:any[]) => { if(!els?.length) return; const i=els[0].index; const label=(chartHours.current as any)?.data?.labels?.[i]; if(label) nav(`/?q=${encodeURIComponent('hour:'+label)}`) }
        }
      })
    }
    const meta = data.meta_presence || []
    if (refMeta.current) {
      chartMeta.current?.destroy()
      chartMeta.current = new Chart(refMeta.current, {
        type:'bar',
        data:{ labels: meta.map((x:any)=>x[0]), datasets:[{ label:'Заполненность метаданных', data: meta.map((x:any)=>x[1]), backgroundColor: palette.gray }] },
        options:{ ...hbarOptions, indexAxis:'y' }
      })
    }
    const avg = data.avg_size_type || []
    if (refAvgSize.current) {
      chartAvg.current?.destroy()
      chartAvg.current = new Chart(refAvgSize.current, {
        type:'bar',
        data:{ labels: avg.map((x:any)=>x[0]), datasets:[{ label:'Средний размер (МБ)', data: avg.map((x:any)=>x[1]), backgroundColor: palette.green }] },
        options:{ ...commonOptions }
      })
    }
    // Распределение файлов по коллекциям
    const coll = data.collections_counts || []
    if (refCollections.current) {
      chartCollections.current?.destroy()
      chartCollections.current = new Chart(refCollections.current, {
        type:'bar',
        data:{ labels: coll.map((x:any)=>x[0]), datasets:[{ label:'По коллекциям', data: coll.map((x:any)=>x[1]), backgroundColor: palette.accent }] },
        options:{ ...hbarOptions, indexAxis:'y' }
      })
    }
    // Совокупный размер коллекций (в МБ)
    const collSize = (data.collections_total_size||[]).map(([name, bytes]) => [name, (Number(bytes||0)/(1024*1024))] as any)
    if (refCollSize.current) {
      chartCollSize.current?.destroy()
      chartCollSize.current = new Chart(refCollSize.current, {
        type:'bar',
        data:{ labels: collSize.map((x:any)=>x[0]), datasets:[{ label:'Объём (МБ)', data: collSize.map((x:any)=>Math.round(Number(x[1])*10)/10), backgroundColor: palette.primary }] },
        options:{ ...hbarOptions, indexAxis:'y' }
      })
    }
    // Самые крупные файлы (в МБ)
    const largest = (data.largest_files||[]).map(([label, bytes]) => [label, (Number(bytes||0)/(1024*1024))] as any)
    if (refLargest.current) {
      chartLargest.current?.destroy()
      chartLargest.current = new Chart(refLargest.current, {
        type:'bar',
        data:{ labels: largest.map((x:any)=>x[0]), datasets:[{ label:'Размер (МБ)', data: largest.map((x:any)=>Math.round(Number(x[1])*10)/10), backgroundColor: palette.red }] },
        options:{ ...hbarOptions, indexAxis:'y' }
      })
    }
    // Ключи тегов
    const tagKeys = data.tag_keys || []
    if (!selectedKey && tagKeys.length>0) setSelectedKey(String(tagKeys[0][0]||''))
    if (refTagKeys.current) {
      chartTagKeys.current?.destroy()
      chartTagKeys.current = new Chart(refTagKeys.current, {
        type:'bar', data:{ labels: tagKeys.map((x:any)=>tagKeyRu(String(x[0]||''))), datasets:[{ label:'Ключи тегов', data: tagKeys.map((x:any)=>x[1]), backgroundColor: palette.gray }] },
        options:{ ...hbarOptions, indexAxis:'y', onClick: (_evt:any, els:any[])=>{ if(!els?.length) return; const i=els[0].index; const raw = tagKeys?.[i]?.[0]; if (raw) setSelectedKey(String(raw)) } }
      })
    }
    // Горизонтальная диаграмма по топовым авторам
    const authors = data.authors || []
    if (refAuthorsBar.current) {
      chartAuthorsBar.current?.destroy()
      chartAuthorsBar.current = new Chart(refAuthorsBar.current, {
        type:'bar',
        data:{ labels: authors.map((x:any)=>x[0]), datasets:[{ label:'Топ авторов', data: authors.map((x:any)=>x[1]), backgroundColor: palette.yellow }] },
        options:{ ...hbarOptions, indexAxis:'y', onClick: (_evt:any, els:any[])=>{ if(!els?.length) return; const i=els[0].index; const label = (chartAuthorsBar.current as any)?.data?.labels?.[i]; if (label) nav(`/?q=${encodeURIComponent(String(label))}`) } }
      })
    }
    // При смене темы или данных уничтожаем диаграммы, чтобы избежать утечек
    return () => {
      chartTypes.current?.destroy(); chartMonths.current?.destroy(); chartYears.current?.destroy(); chartSizes.current?.destroy(); chartExts.current?.destroy(); chartWeekdays.current?.destroy(); chartHours.current?.destroy(); chartMeta.current?.destroy(); chartAvg.current?.destroy(); chartAuthorsBar.current?.destroy(); chartCollections.current?.destroy(); chartCollSize.current?.destroy(); chartLargest.current?.destroy(); chartTagKeys.current?.destroy(); chartTagVals.current?.destroy()
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, palette, rangeMonths, cumulativeMonths, logScale])

  function exportCSV(name: string, rows: StatKV[]) {
    const header = 'label,value\n'
    const body = rows.map(([k,v]) => `${String(k).replaceAll('"','""')},${v}`).join('\n')
    const csv = header + body
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const a = document.createElement('a')
    const url = URL.createObjectURL(blob)
    a.href = url
    a.download = `${name}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  useEffect(()=>{
    let ignore=false
    ;(async()=>{
      if (!selectedKey) { setTagVals([]); return }
      try{
        const r = await fetch(`/api/stats/tag-values?key=${encodeURIComponent(selectedKey)}&limit=30`)
        const j = await r.json(); if (!ignore){ setTagVals(j.items||[]) }
      }catch{ if (!ignore) setTagVals([]) }
    })()
    return ()=>{ ignore=true }
  }, [selectedKey])

  useEffect(()=>{
    if (!refTagVals.current) return
    chartTagVals.current?.destroy()
    chartTagVals.current = new Chart(refTagVals.current, {
      type:'bar', data:{ labels: tagVals.map(x=>x[0]), datasets:[{ label:`Значения тега: ${tagKeyRu(selectedKey || '')}`, data: tagVals.map(x=>x[1]), backgroundColor: '#4e79a7' }] },
      options:{ responsive:true, maintainAspectRatio:false, indexAxis:'y', scales:{ x:{ type: logScale? 'logarithmic':'linear' } } }
    })
  }, [tagVals, selectedKey, logScale])

  const totalInfo = useMemo(() => {
    if (!data) return null
    const bytes = Number(data.total_size_bytes||0)
    const units = ['Б','КБ','МБ','ГБ','ТБ']
    let v = bytes; let i = 0
    while (v >= 1024 && i < units.length-1) { v/=1024; i++ }
    return { files: numberFormatter.format(data.total_files||0), size: `${v.toFixed(1)} ${units[i]}` }
  }, [data, numberFormatter])

  const recentInfo = useMemo(() => {
    if (!data?.recent_counts) return null
    return {
      '30d': numberFormatter.format(data.recent_counts['30d'] || 0),
      '7d': numberFormatter.format(data.recent_counts['7d'] || 0),
    }
  }, [data, numberFormatter])

  const tagsInfo = useMemo(() => {
    const summary = data?.tags_summary
    if (!summary) return null
    return {
      avg: (Number(summary.avg_per_file || 0)).toFixed(1),
      withTags: numberFormatter.format(summary.with_tags || 0),
      withoutTags: numberFormatter.format(summary.without_tags || 0),
    }
  }, [data, numberFormatter])

  const metaMissing = useMemo(() => {
    if (!data?.meta_missing) return []
    return [...data.meta_missing]
      .filter(([, count]) => Number(count || 0) > 0)
      .sort((a, b) => Number(b?.[1] || 0) - Number(a?.[1] || 0))
  }, [data])
  if (error && !data) {
    return <div className="card p-3 text-danger">Не удалось загрузить статистику: {error}</div>
  }

  if (loading && !data) {
    return (
      <div className="card p-3">
        <LoadingState title="Загружаем статистику" description="Готовим диаграммы и агрегаты по библиотеке" lines={6} />
      </div>
    )
  }

  return (
    <div className="card p-3 stats-page-glass" aria-busy={loading}>
      <div className="d-flex align-items-center justify-content-between mb-2">
        <div className="fw-semibold">Статистика</div>
        <div className="d-flex align-items-center gap-3">
          <label className="form-label m-0">Период:</label>
          <select className="form-select form-select-sm" value={rangeMonths} onChange={e=>setRangeMonths(parseInt(e.target.value))} style={{width:140}}>
            <option value={12}>12 мес</option>
            <option value={24}>24 мес</option>
            <option value={36}>36 мес</option>
            <option value={0}>Все</option>
          </select>
          <div className="form-check form-switch m-0">
            <input id="logscale" className="form-check-input" type="checkbox" checked={logScale} onChange={e=>setLogScale(e.target.checked)} />
            <label htmlFor="logscale" className="form-check-label">Лог‑шкала</label>
          </div>
          <div className="form-check form-switch m-0">
            <input id="cum" className="form-check-input" type="checkbox" checked={cumulativeMonths} onChange={e=>setCumulativeMonths(e.target.checked)} />
            <label htmlFor="cum" className="form-check-label">Накопит.</label>
          </div>
        </div>
      </div>
      {loading && (
        <div className="row g-3" role="status" aria-live="polite">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="col-12 col-xl-4">
              <LoadingState variant="card" lines={4} />
            </div>
          ))}
        </div>
      )}
      {error && <div className="text-danger">Не удалось обновить данные: {error}</div>}
      {!loading && data && data.total_files > 0 && (
        <>
          <div className="row g-3">
            <div className="col-6 col-xl-3">
              <div className="card p-3 text-center kpi" aria-labelledby="kpi-files">
                <div id="kpi-files" className="label muted">Файлов</div>
                <div className="value">{totalInfo?.files}</div>
              </div>
            </div>
            <div className="col-6 col-xl-3">
              <div className="card p-3 text-center kpi" aria-labelledby="kpi-size">
                <div id="kpi-size" className="label muted">Размер</div>
                <div className="value">{totalInfo?.size}</div>
              </div>
            </div>
            <div className="col-6 col-xl-3">
              <div className="card p-3 text-center kpi" aria-labelledby="kpi-recent">
                <div id="kpi-recent" className="label muted">Новых за 30 дней</div>
                <div className="value">{recentInfo?.['30d'] || '0'}</div>
                <div className="muted" style={{ fontSize: 12 }}>За 7 дн: {recentInfo?.['7d'] || '0'}</div>
              </div>
            </div>
            <div className="col-6 col-xl-3">
              <div className="card p-3 text-center kpi" aria-labelledby="kpi-tags">
                <div id="kpi-tags" className="label muted">Среднее тегов</div>
                <div className="value">{tagsInfo?.avg || '0.0'}</div>
                <div className="muted" style={{ fontSize: 12 }}>С тегами: {tagsInfo?.withTags || '0'}</div>
                <div className="muted" style={{ fontSize: 12 }}>Без тегов: {tagsInfo?.withoutTags || '0'}</div>
              </div>
            </div>
          </div>
          {!loading && data?.feedback_summary && (Number(data.feedback_summary.negative || 0) > 0) && (
            <div className="alert alert-warning mt-3" role="alert">
              Несколько документов получили отрицательный вес по пользовательскому фидбэку. Проверьте список «Низкие оценки пользователей» или страницу «AI метрики» для корректирующих действий.
            </div>
          )}
          <div className="row g-3 mt-1">
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">По типам</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('types', data?.types || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refTypes}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">По месяцам</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> {
                    const monthsRaw = (data?.months||[]).slice(rangeMonths>0? -rangeMonths: undefined)
                    const rows = cumulativeMonths ? (function(){ let acc=0; return monthsRaw.map(([m,v]: any) => { acc += Number(v||0); return [m, acc] as any }) })() : monthsRaw
                    exportCSV('months', rows)
                  }}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refMonths}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-4">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Размеры файлов</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('sizes', data?.sizes || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refSizes}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-4">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">По расширениям</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('exts', data?.exts || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refExts}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-4">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Дни недели</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('weekdays', data?.weekdays || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refWeekdays}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-4">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Часы</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('hours', data?.hours || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refHours}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Заполненность метаданных</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('meta_presence', data?.meta_presence || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refMeta}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-3">
                <div className="fw-semibold mb-2">Поля без заполнения</div>
                <div className="d-flex flex-column gap-1">
                  {metaMissing.length === 0 && <div className="text-muted">Все данные заполнены</div>}
                  {metaMissing.map(([label, count]) => (
                    <div key={label} className="d-flex justify-content-between" style={{ fontSize: 13 }}>
                      <span>{label}</span>
                      <span className="text-muted">{numberFormatter.format(count)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Средний размер (МБ)</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('avg_size_by_type', data?.avg_size_type || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refAvgSize}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">По годам</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('years', data?.years || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refYears}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Топ авторов</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('authors', data?.authors || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refAuthorsBar}/></div>
              </div>
            </div>
          </div>
          <div className="row g-3 mt-1">
            <div className="col-12 col-xl-6">
              <div className="card p-3">
                <div className="fw-semibold mb-2">Облако авторов</div>
                <div style={{minHeight:200}}>
                  {(data.authors_cloud||[]).map((x:any,i:number)=>{
                    const [name,cnt]=x; const size=12+Math.min(26, Math.round(Math.log(1+cnt)*8));
                    return (
                      <button key={i} className="btn btn-link p-0 me-2" style={{fontSize:size, lineHeight:'2.2rem'}} onClick={()=> nav(`/?q=${encodeURIComponent(String(name))}`)}>{name}</button>
                    )
                  })}
                </div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-3">
                <div className="fw-semibold mb-2">Облако ключевых слов</div>
                <div style={{minHeight:200}}>
                  {(data.top_keywords||[]).map((x:any,i:number)=>{
                    const [val,cnt]=x; const size=12+Math.min(24, Math.round(Math.log(1+cnt)*7));
                    return (
                      <button key={i} className="btn btn-link p-0 me-2" style={{fontSize:size, lineHeight:'2.2rem'}} onClick={()=> nav(`/?q=${encodeURIComponent(String(val))}`)}>{val}</button>
                    )
                  })}
                </div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">По коллекциям</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('collections', data?.collections_counts || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refCollections}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Ключи тегов</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('tag_keys', data?.tag_keys || [])}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refTagKeys}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Значения тега ({tagKeyRu(selectedKey || '')})</div>
                  <div className="d-flex align-items-center gap-2">
                    <select className="form-select form-select-sm" style={{width:180}} value={selectedKey} onChange={e=>setSelectedKey(e.target.value)}>
                      {(data?.tag_keys||[]).map((x:any,i:number)=> <option key={i} value={String(x[0]||'')}>{tagKeyRu(String(x[0]||''))}</option>)}
                    </select>
                    <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV(`tag_values_${selectedKey}`, tagVals || [])}>CSV</button>
                  </div>
                </div>
                <div className="chart-body"><canvas ref={refTagVals}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Объём по коллекциям (МБ)</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('collections_total_size_mb', (data?.collections_total_size||[]).map(([n,b]:any)=>[n, Math.round(Number(b||0)/(1024*1024))]))}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refCollSize}/></div>
              </div>
            </div>
            <div className="col-12 col-xl-6">
              <div className="card p-2 chart-card">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="fw-semibold chart-title">Самые большие файлы (МБ)</div>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=> exportCSV('largest_files_mb', (data?.largest_files||[]).map(([n,b]:any)=>[n, Math.round(Number(b||0)/(1024*1024))]))}>CSV</button>
                </div>
                <div className="chart-body"><canvas ref={refLargest}/></div>
              </div>
            </div>
          </div>
          {(Boolean(data.feedback_positive?.length) || Boolean(data.feedback_negative?.length) || data.feedback_summary) && (
            <div className="row g-3 mt-1">
              {Boolean(data.feedback_positive?.length) && (
                <div className="col-12 col-xl-6 col-xxl-4">
                  <div className="card p-3 h-100">
                    <div className="fw-semibold mb-2">Документы с лучшим откликом</div>
                    <ol className="mb-0 ps-3">
                      {(data.feedback_positive || []).slice(0, 10).map(entry => (
                        <li key={entry.file_id} className="mb-2">
                          <button className="btn btn-link p-0 text-start" style={{ fontSize: 14 }} onClick={()=> nav(`/?q=${encodeURIComponent(entry.title || '')}`)}>
                            {entry.title}
                          </button>
                          <div className="text-secondary" style={{ fontSize: 12 }}>
                            {(entry.author || '').trim() ? `${entry.author} · ` : ''}{entry.year || '—'} · вес {formatScore(entry.weight)}
                          </div>
                          <div className="text-secondary" style={{ fontSize: 12 }}>
                            👍 {entry.positive} · 👎 {entry.negative} · кликов {entry.clicks}
                          </div>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>
              )}
              {Boolean(data.feedback_negative?.length) && (
                <div className="col-12 col-xl-6 col-xxl-4">
                  <div className="card p-3 h-100">
                    <div className="fw-semibold mb-2">Низкие оценки пользователей</div>
                    <ol className="mb-0 ps-3">
                      {(data.feedback_negative || []).slice(0, 10).map(entry => (
                        <li key={entry.file_id} className="mb-2">
                          <button className="btn btn-link p-0 text-start" style={{ fontSize: 14 }} onClick={()=> nav(`/?q=${encodeURIComponent(entry.title || '')}`)}>
                            {entry.title}
                          </button>
                          <div className="text-secondary" style={{ fontSize: 12 }}>
                            {(entry.author || '').trim() ? `${entry.author} · ` : ''}{entry.year || '—'} · вес {formatScore(entry.weight)}
                          </div>
                          <div className="text-secondary" style={{ fontSize: 12 }}>
                            👍 {entry.positive} · 👎 {entry.negative} · кликов {entry.clicks}
                          </div>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>
              )}
              {data.feedback_summary && (
                <div className="col-12 col-xl-6 col-xxl-4">
                  <div className="card p-3 h-100">
                    <div className="fw-semibold mb-2">Сводка фидбэка</div>
                    <div className="d-flex flex-column gap-1" style={{ fontSize: 13 }}>
                      <div className="d-flex justify-content-between"><span>Всего файлов с весами</span><span className="text-secondary">{numberFormatter.format(data.feedback_summary.total_files || 0)}</span></div>
                      <div className="d-flex justify-content-between"><span>Положительный вес</span><span className="text-success">{numberFormatter.format(data.feedback_summary.positive || 0)}</span></div>
                      <div className="d-flex justify-content-between"><span>Отрицательный вес</span><span className="text-danger">{numberFormatter.format(data.feedback_summary.negative || 0)}</span></div>
                      <div className="text-secondary" style={{ fontSize: 12 }}>Вес рассчитывается из оценок релевантности и кликов (`POST /api/ai-search/feedback`).</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
      {!loading && data && data.total_files === 0 && (
        <div className="mt-3">
          <EmptyState title="Нет данных для статистики" description="Добавьте файлы в каталог, чтобы увидеть метрики." />
        </div>
      )}
    </div>
  )
}
