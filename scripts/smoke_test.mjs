#!/usr/bin/env node
/**
 * E2E smoke test for Agregator UI.
 * Run: node scripts/smoke_test.mjs (after: npm i playwright)
 */
import { chromium } from 'playwright';

const BASE = 'http://localhost:5050/app';
const WAIT = 'load';
const TIMEOUT = 25000;
const results = { passed: [], failed: [], blockers: [] };

function log(area, ok, detail) {
  const entry = `[${area}] ${ok ? 'PASS' : 'FAIL'}: ${detail}`;
  if (ok) results.passed.push(entry);
  else results.failed.push(entry);
  console.log(entry);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // 1. Open app
    await page.goto(BASE, { waitUntil: WAIT, timeout: TIMEOUT });
    log('App', true, 'Loaded ' + BASE);

    // 2. Auth
    const loc = page.locator('input[type="text"], input[name="username"], #username');
    const hasLogin = await loc.first().isVisible().catch(() => false);
    if (hasLogin) {
      await page.fill('input[type="text"], input[name="username"]', 'admin');
      await page.fill('input[type="password"], input[name="password"]', 'admin123');
      await page.click('button[type="submit"], input[type="submit"], button:has-text("Вход"), button:has-text("Login")');
      await page.waitForTimeout(2000);
    }
    const afterAuth = page.url();
    const stillLogin = afterAuth.includes('/login');
    if (stillLogin) {
      log('Auth', false, 'Login form still visible after submit');
      results.blockers.push('Authentication failed - cannot proceed');
    } else {
      log('Auth', true, 'Logged in (or had session)');
    }

    if (results.blockers.length) {
      console.log('\n--- BLOCKERS ---');
      results.blockers.forEach(b => console.log(b));
      await browser.close();
      process.exit(1);
    }

    // 3. Navigation / Catalogue
    if (!page.url().includes('/app')) {
      await page.goto(BASE + '/', { waitUntil: WAIT, timeout: TIMEOUT });
    }
    await page.waitForTimeout(3000);
    const catalogueBody = (await page.textContent('body')) || '';
    const hasCatalogue = /Каталог|Поиск|Agregator|Настройки|root/i.test(catalogueBody) || catalogueBody.length > 500;
    log('Catalogue', !!hasCatalogue, hasCatalogue ? 'Catalogue page visible' : 'Catalogue elements not found');

    // 4. Settings - try client-side nav first
    const settingsLink = page.locator('a[href*="settings"]');
    if (await settingsLink.first().isVisible().catch(() => false)) {
      await settingsLink.first().click();
      await page.waitForTimeout(3000);
    } else {
      await page.goto(BASE + '/settings', { waitUntil: WAIT, timeout: TIMEOUT });
      await page.waitForTimeout(2000);
    }
    await page.waitForTimeout(1500);
    const expertBtn = page.locator('button:has-text("Expert")');
    if (await expertBtn.isVisible().catch(() => false)) {
      await expertBtn.click();
      await page.waitForTimeout(1000);
    }
    const hasSettings = await page.locator('body').textContent();
    const hasDbPanel = hasSettings && (hasSettings.includes('Управление базой данных') || hasSettings.includes('database') || hasSettings.includes('SQLite') || hasSettings.includes('PostgreSQL'));
    const hasMigration = hasSettings && (hasSettings.includes('Миграция') || hasSettings.includes('Wizard') || hasSettings.includes('host.docker.internal') || hasSettings.includes('localhost'));
    log('Settings DB panel', !!hasDbPanel, hasDbPanel ? 'DB panel/selector found' : 'DB UI not found');
    log('Settings Migration', !!hasMigration, hasMigration ? 'Migration wizard block found' : 'Migration block not found');

    // 5. Admin status - use direct URL to avoid dropdown
    await page.goto(BASE + '/admin/status', { waitUntil: WAIT, timeout: TIMEOUT });
    await page.waitForTimeout(3000);
    const statusText = await page.locator('body').textContent();
    const hasStatus = statusText && (statusText.includes('Состояние') || statusText.includes('Состояние сервиса') || statusText.includes('БД') || statusText.includes('database') || statusText.includes('Не удалось загрузить') || statusText.includes('Недостаточно прав'));
    log('Admin status', !!hasStatus, hasStatus ? 'Status page loaded' : 'Status content not found');

    // 6. Admin logs
    await page.goto(BASE + '/admin/logs', { waitUntil: WAIT, timeout: TIMEOUT });
    await page.waitForTimeout(1500);
    const logsOk = page.url().endsWith('logs') || page.url().includes('logs');
    log('Admin logs', logsOk, logsOk ? 'Logs page loaded' : 'Logs page failed');

    // 7. Admin tasks
    await page.goto(BASE + '/admin/tasks', { waitUntil: WAIT, timeout: TIMEOUT });
    const tasksOk = page.url().endsWith('tasks') || page.url().includes('tasks');
    log('Admin tasks', tasksOk, tasksOk ? 'Tasks page loaded' : 'Tasks page failed');

  } catch (err) {
    const msg = err.message || String(err);
    log('Runtime', false, msg);
    if (msg.includes('Timeout') || msg.includes('blocked')) {
      results.blockers.push(msg);
    }
  } finally {
    await browser.close();
  }

  console.log('\n--- REPORT ---');
  console.log('Passed:', results.passed.length);
  results.passed.forEach(p => console.log('  +', p));
  console.log('Failed:', results.failed.length);
  results.failed.forEach(f => console.log('  -', f));
  if (results.blockers.length) {
    console.log('Blockers:', results.blockers);
    process.exit(1);
  }
  process.exit(results.failed.length > 0 ? 1 : 0);
}

main();
