export function materialTypeRu(type: string | null | undefined, fallback: string = 'Другое'): string {
  if (!type) return fallback
  const map: Record<string, string> = {
    dissertation: 'Диссертация',
    dissertation_abstract: 'Автореферат',
    article: 'Статья',
    textbook: 'Учебник',
    monograph: 'Монография',
    report: 'Отчёт',
    patent: 'Патент',
    presentation: 'Презентация',
    proceedings: 'Труды',
    standard: 'Стандарт',
    note: 'Заметка',
    document: 'Документ',
    file: 'Файл',
    audio: 'Аудио',
    image: 'Изображение',
    video: 'Видео',
    other: 'Другое',
  }
  const key = String(type).toLowerCase()
  return map[key] || String(type)
}

export function tagKeyRu(key: string | null | undefined): string {
  if (!key) return '—'
  const map: Record<string, string> = {
    lang: 'Язык',
    ext: 'Расширение',
    pages: 'Страниц',
    doi: 'DOI',
    isbn: 'ISBN',
    journal: 'Журнал',
    volume_issue: 'Том/номер',
    number: 'Номер',
    pages_range: 'Страницы',
    organization: 'Организация',
    organizations: 'Организации',
    conference: 'Конференция',
    author: 'Автор',
    keyword: 'Ключевое слово',
    topic: 'Тема',
    category: 'Категория',
    classification: 'Классификация',
    specialty: 'Специальность',
    speciality: 'Специальность',
    speciality_code: 'Код специальности',
    supervisor: 'Научный руководитель',
    advisor: 'Научный руководитель',
    consultant: 'Консультант',
    reviewer: 'Рецензент',
    degree: 'Степень',
    faculty: 'Факультет',
    department: 'Кафедра',
    institute: 'Институт',
    university: 'Университет',
    subject: 'Предмет',
    discipline: 'Дисциплина',
    direction: 'Направление',
    project: 'Проект',
    grant: 'Грант',
    funding: 'Финансирование',
    speciality_group: 'Группа специальностей',
    specialty_group: 'Группа специальностей',
    speciality_code_ru: 'Код специальности (РУ)',
    specialty_code_ru: 'Код специальности (РУ)',
    speciality_code_en: 'Код специальности (EN)',
    specialty_code_en: 'Код специальности (EN)',
    speciality_name: 'Название специальности',
    specialty_name: 'Название специальности',
    speciality_field: 'Область специальности',
    specialty_field: 'Область специальности',
    speciality_profile: 'Профиль специальности',
    specialty_profile: 'Профиль специальности',
    speciality_level: 'Уровень специальности',
    specialty_level: 'Уровень специальности',
    speciality_form: 'Форма обучения',
    specialty_form: 'Форма обучения',
    speciality_mode: 'Форма обучения',
    specialty_mode: 'Форма обучения',
    specialization: 'Специализация',
    qualification: 'Квалификация',
    education_form: 'Форма обучения',
    education_level: 'Уровень образования',
    education_program: 'Образовательная программа',
    scientific_degree: 'Ученая степень',
    scientific_title: 'Учёное звание',
    scientific_specialty: 'Научная специальность',
    speciality_code_vak: 'Код специальности ВАК',
    specialty_code_vak: 'Код специальности ВАК',
    vak: 'Код ВАК',
    udk: 'УДК',
    specialty_code: 'Код специальности',
    specialty_index: 'Индекс специальности',
    specialty_group_code: 'Код группы специальностей',
    speciality_group_code: 'Код группы специальностей',
    speciality_direction: 'Направление подготовки',
    specialty_direction: 'Направление подготовки',
  }
  const keyLower = key.toLowerCase()
  return map[keyLower] || key
}

export function taskStatusRu(status: string | null | undefined): string {
  if (!status) return '—'
  const map: Record<string, string> = {
    running: 'Выполняется',
    queued: 'В очереди',
    pending: 'Ожидает',
    cancelling: 'Отменяется',
    cancelled: 'Отменена',
    completed: 'Завершена',
    error: 'Ошибка',
    success: 'Успешно',
  }
  const key = status.toLowerCase()
  return map[key] || status
}

export function actionLogLevelRu(level: 'info' | 'error' | 'success'): string {
  const map: Record<typeof level, string> = {
    info: 'ИНФО',
    error: 'ОШИБКА',
    success: 'УСПЕХ',
  }
  return map[level]
}
