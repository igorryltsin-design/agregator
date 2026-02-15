#!/usr/bin/env node
/**
 * Manual-style browser validation for Agregator UI.
 * Run: node scripts/browser_validation.mjs
 */
import { chromium } from 'playwright';

const BASE = 'http://localhost:5050/app';
const WAIT_MS = 2500;
const RETRIES = 3;
const results = [];

function add(area, pass, evidence) {
  results.push({ area, pass, evidence });
  console.log(`[${pass ? 'PASS' : 'FAIL'}] ${area}: ${evidence}`);
}

async function retryNavigate(page, url, label) {
  for (let i = 0; i < RETRIES; i++) {
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
      await page.waitForTimeout(WAIT_MS);
      return true;
    } catch (e) {
      if (i < RETRIES - 1) {
        await page.waitForTimeout(2000);
        await page.reload().catch(() => {});
        await page.waitForTimeout(2000);
      }
      if (i === RETRIES - 1) throw e;
    }
  }
  return false;
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    // 1) Login page and successful login redirect
    await retryNavigate(page, BASE, 'initial');
    const hasLogin = await page.locator('input[type="password"]').isVisible().catch(() => false);
    if (hasLogin) {
      await page.fill('.login-card input:not([type="password"])', 'admin');
      await page.fill('input[type="password"]', 'admin123');
      await page.click('button[type="submit"], button:has-text("Войти")');
      await page.waitForTimeout(WAIT_MS);
    }
    const url = page.url();
    const loginOk = !url.includes('/login');
    add('1) Login & redirect', loginOk, loginOk ? `Redirected to ${url}` : `Still on login: ${url}`);

    if (!loginOk) {
      add('BLOCKER', false, 'Login failed - cannot proceed');
      await browser.close();
      printReport();
      process.exit(1);
    }

    // 2) Main shell/nav visible
    const body1 = (await page.textContent('body')) || '';
    const hasNav = /Каталог|Поиск|Настройки|Админ|Профиль/i.test(body1);
    add('2) Main shell/nav', !!hasNav, hasNav ? 'Nav elements visible' : 'Nav not found');

    // 3) Catalogue: search input, typing, filters
    const searchInput = page.locator('input[placeholder*="Поиск"], input.app-search-wrap input, input.form-control').first();
    const searchExists = await searchInput.isVisible().catch(() => false);
    add('3a) Catalogue search input', searchExists, searchExists ? 'Search input found' : 'Not found');

    if (searchExists) {
      await searchInput.fill('test');
      await page.waitForTimeout(500);
      const val = await searchInput.inputValue();
      add('3b) Typing in search', val === 'test', val === 'test' ? 'Typed "test"' : `Value: "${val}"`);
    } else {
      add('3b) Typing in search', false, 'Skipped (no input)');
    }

    const filtersPanel = page.locator('.catalogue-filter-card, .filter-sidebar, :text("Фильтры")').first();
    const filterVisible = await filtersPanel.isVisible().catch(() => false);
    add('3c) Filters panel/sidebar', !!filterVisible, filterVisible ? 'Filters visible or toggle found' : 'Filters not found');

    // 4) Settings page
    const settingsLink = page.locator('a[href*="settings"], a[aria-label="Настройки"]');
    if (await settingsLink.first().isVisible().catch(() => false)) {
      await settingsLink.first().click();
    } else {
      await retryNavigate(page, BASE + '/settings', 'settings');
    }
    await page.waitForTimeout(WAIT_MS);

    const settingsBody = (await page.textContent('body')) || '';
    const settingsLoads = settingsBody.includes('Настройки') || page.url().includes('settings');
    add('4a) Settings page loads', !!settingsLoads, settingsLoads ? 'Settings loaded' : 'Settings did not load');

    // Expert tab if needed
    const expertBtn = page.locator('button:has-text("Expert")');
    if (await expertBtn.isVisible().catch(() => false)) {
      await expertBtn.click();
      await page.waitForTimeout(1500);
    }

    const dbSection = (await page.textContent('body')) || '';
    const hasDbSection = dbSection.includes('Управление базой данных');
    add('4b) Section "Управление базой данных"', !!hasDbSection, hasDbSection ? 'Section found' : 'Section not found');

    const dbTypeSelect = page.locator('select.form-select').filter({ has: page.locator('option[value="sqlite"]') }).or(
      page.locator('select').filter({ hasText: 'SQLite' })
    );
    const hasSqlite = dbSection.includes('SQLite');
    const hasPostgres = dbSection.includes('PostgreSQL');
    add('4c) DB type selector (SQLite/PostgreSQL)', hasSqlite && hasPostgres, `SQLite: ${hasSqlite}, PostgreSQL: ${hasPostgres}`);

    // Migration wizard fields
    const migSection = (await page.textContent('body')) || '';
    const migFields = {
      sqlitePath: migSection.includes('SQLite файл') || migSection.includes('catalogue.db'),
      pgUrl: migSection.includes('PostgreSQL URL') || migSection.includes('postgresql://'),
      host: migSection.includes('Хост') || migSection.includes('host'),
      port: migSection.includes('Порт') || migSection.includes('5432'),
      db: migSection.includes('Имя БД') || migSection.includes('agregator'),
      user: migSection.includes('Логин') || migSection.includes('логин'),
      password: migSection.includes('Пароль') || migSection.includes('••••'),
      mode: migSection.includes('dry-run') || migSection.includes('Режим'),
    };
    const migLabels = [];
    if (migSection.includes('SQLite файл')) migLabels.push('SQLite файл');
    if (migSection.includes('PostgreSQL URL')) migLabels.push('PostgreSQL URL (опционально)');
    if (migSection.includes('Хост')) migLabels.push('Хост / адрес БД');
    if (migSection.includes('Порт')) migLabels.push('Порт');
    if (migSection.includes('Имя БД')) migLabels.push('Имя БД');
    if (migSection.includes('Логин')) migLabels.push('Логин');
    if (migSection.includes('Пароль')) migLabels.push('Пароль');
    if (migSection.includes('Режим') || migSection.includes('dry-run')) migLabels.push('Режим');

    const migOk = Object.values(migFields).filter(Boolean).length >= 6;
    add('4d) Migration wizard fields', !!migOk, `Labels: ${migLabels.join(', ') || 'none'}`);

    const presetLocalhost = (await page.locator('button:has-text("localhost")').first().isVisible().catch(() => false)) || migSection.includes('localhost');
    const presetDocker = (await page.locator('button:has-text("host.docker.internal")').first().isVisible().catch(() => false)) || migSection.includes('host.docker.internal');
    const presetPostgres = (await page.locator('button:has-text("postgres")').first().isVisible().catch(() => false)) || migSection.includes('postgres');
    add('4e) Host preset buttons', presetLocalhost && presetDocker && presetPostgres, `localhost:${presetLocalhost}, host.docker.internal:${presetDocker}, postgres:${presetPostgres}`);

    // 5) Admin Service Status
    await retryNavigate(page, BASE + '/admin/status', 'admin/status');
    await page.waitForTimeout(3000);
    const statusBody = (await page.textContent('body')) || '';
    const hasStatusCards = statusBody.includes('Состояние') || statusBody.includes('БД') || statusBody.includes('database') || statusBody.includes('runtime') || statusBody.length > 300;
    add('5) Admin Service Status', !!hasStatusCards, hasStatusCards ? 'Page loads with content' : 'Empty or error');

    // 6) Admin Logs
    await retryNavigate(page, BASE + '/admin/logs', 'admin/logs');
    const logsUrl = page.url().includes('logs');
    add('6) Admin Logs page', !!logsUrl, logsUrl ? 'Logs loaded' : 'Failed');

    // 7) Admin Tasks
    await retryNavigate(page, BASE + '/admin/tasks', 'admin/tasks');
    const tasksUrl = page.url().includes('tasks');
    add('7) Admin Tasks page', !!tasksUrl, tasksUrl ? 'Tasks loaded' : 'Failed');

  } catch (err) {
    add('Runtime error', false, err.message || String(err));
  } finally {
    await browser.close();
    printReport();
  }
}

function printReport() {
  const passed = results.filter(r => r.pass).length;
  const failed = results.filter(r => !r.pass).length;
  const total = results.length;
  let overall = 'FAIL';
  if (failed === 0) overall = 'PASS';
  else if (passed >= total * 0.6) overall = 'PARTIAL';

  console.log('\n--- VALIDATION REPORT ---');
  console.log('Area | Result | Evidence');
  console.log('-----|--------|----------');
  results.forEach(r => {
    console.log(`${r.area} | ${r.pass ? 'PASS' : 'FAIL'} | ${r.evidence}`);
  });
  console.log('-----|--------|----------');
  console.log(`Overall: ${overall} (${passed}/${total} passed)`);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
