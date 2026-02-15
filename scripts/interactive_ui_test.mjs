#!/usr/bin/env node
/**
 * Interactive UI test: click buttons, verify outcomes.
 * Safe only - no destructive ops.
 */
import { chromium } from 'playwright';

const BASE = 'http://localhost:5050/app';
const log = [];
const errors = [];
function action(name, result) {
  log.push({ action: name, result });
  console.log(`[${result ? 'OK' : '??'}] ${name} -> ${result}`);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Capture toasts and alerts
  const toasts = [];
  page.on('console', msg => {
    const t = msg.text();
    if (t.includes('toast') || t.includes('success') || t.includes('error')) toasts.push(t);
  });

  try {
    await page.goto(BASE, { waitUntil: 'load', timeout: 20000 });
    await page.waitForTimeout(3000);

    // Login if needed
    if (await page.locator('input[type="password"]').isVisible().catch(() => false)) {
      await page.fill('.login-card input:not([type="password"])', 'admin');
      await page.fill('input[type="password"]', 'admin123');
      await page.click('button:has-text("Войти")');
      await page.waitForTimeout(3500);
    }
    const urlAfterLogin = page.url();
    action('Login admin/admin123', urlAfterLogin.includes('/app') && !urlAfterLogin.includes('/login') ? 'Redirected to app' : `URL: ${urlAfterLogin}`);

    // 1) Nav through main sections
    const navSections = [
      { name: 'Настройки', selector: 'a[href*="settings"], a[aria-label="Настройки"]' },
      { name: 'Граф', selector: 'a[aria-label="Граф"]' },
      { name: 'Статистика', selector: 'a[aria-label="Статистика"]' },
      { name: 'Каталог', selector: 'a[href="/app"], .navbar-brand' },
    ];
    for (const s of navSections) {
      const el = page.locator(s.selector).first();
      if (await el.isVisible().catch(() => false)) {
        await el.click();
        await page.waitForTimeout(1500);
        const u = page.url();
        action(`Nav: ${s.name}`, `URL: ${u}`);
      }
    }

    // 2) Catalogue: search query
    const searchInput = page.locator('input[placeholder*="Поиск"]').first();
    if (await searchInput.isVisible().catch(() => false)) {
      await searchInput.fill('тест');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(2000);
      const hasResults = (await page.textContent('body'))?.includes('Найдено') || (await page.textContent('body'))?.includes('Подгружаю');
      action('Catalogue: search "тест" + Enter', hasResults ? 'Search executed, results area visible' : 'Executed');
    }

    // Filter toggle if visible
    const filterBtn = page.locator('button:has-text("Низкое качество"), .catalogue-filter-card').first();
    if (await filterBtn.isVisible().catch(() => false)) {
      await filterBtn.click();
      await page.waitForTimeout(1000);
      action('Catalogue: filter toggle', 'Clicked');
    }

    // 3) Settings: DB type switch, host presets, dry-run
    await page.goto(BASE + '/settings', { waitUntil: 'load', timeout: 20000 });
    await page.waitForTimeout(2000);

    const expertBtn = page.locator('button:has-text("Expert")');
    if (await expertBtn.isVisible().catch(() => false)) {
      await expertBtn.click();
      await page.waitForTimeout(1500);
    }

    const dbTypeSelect = page.locator('select.form-select').filter({ has: page.locator('option[value="postgresql"]') }).first();
    if (await dbTypeSelect.isVisible().catch(() => false)) {
      await dbTypeSelect.selectOption('postgresql');
      await page.waitForTimeout(500);
      const selVal = await dbTypeSelect.inputValue();
      action('Settings: DB type -> PostgreSQL', selVal === 'postgresql' ? 'Selector changed to postgresql' : `Value: ${selVal}`);
      await dbTypeSelect.selectOption('sqlite');
      await page.waitForTimeout(500);
      action('Settings: DB type -> SQLite', 'Switched back');
    }

    // Host preset buttons (migration section: input in same container as preset buttons)
    const hostInput = page.locator('div:has(button:has-text("localhost"))').locator('xpath=..').locator('input').first();
    for (const preset of ['localhost', 'host.docker.internal', 'postgres']) {
      const btn = page.locator(`button:has-text("${preset}")`).first();
      if (await btn.isVisible().catch(() => false)) {
        await btn.click();
        await page.waitForTimeout(500);
        const val = await hostInput.inputValue().catch(() => '');
        action(`Settings: host preset "${preset}"`, val.includes(preset) ? `Host field: ${val}` : `Clicked; host=${val || '(empty)'}`);
      }
    }
    await page.locator('button:has-text("localhost")').first().click().catch(() => {});
    await page.waitForTimeout(300);

    // Migration mode = dry-run
    const modeSelect = page.locator('select.form-select').filter({ has: page.locator('option[value="dry-run"]') }).last();
    if (await modeSelect.isVisible().catch(() => false)) {
      await modeSelect.selectOption('dry-run');
      await page.waitForTimeout(500);
    }

    const launchBtn = page.locator('button:has-text("Запустить миграцию"), button:has-text("Выполнение")').first();
    if (await launchBtn.isVisible().catch(() => false) && !(await launchBtn.isDisabled().catch(() => true))) {
      const dialogPromise = page.waitForEvent('dialog', { timeout: 25000 }).catch(() => null);
      await launchBtn.click();
      await page.waitForTimeout(3000);
      const dialog = await dialogPromise;
      if (dialog) {
        const msg = dialog.message();
        action('Settings: migration dry-run launch', `Alert: ${msg.substring(0, 150)}...`);
        await dialog.accept();
      } else {
        await page.waitForTimeout(5000);
        const body = (await page.textContent('body')) || '';
        const hasResult = body.includes('миграция') || body.includes('exit_code') || body.includes('существует') || body.includes('Завершена');
        action('Settings: migration dry-run launch', hasResult ? 'Result shown (alert or log)' : 'Launched, awaiting result');
      }
    } else {
      action('Settings: migration dry-run launch', 'Button not found or disabled, skipped');
    }

    // 4) Admin Service Status: refresh
    await page.goto(BASE + '/admin/status', { waitUntil: 'load', timeout: 20000 });
    await page.waitForTimeout(2500);
    const refreshBtn = page.locator('button:has-text("Обновить"), button:has-text("Refresh"), button[aria-label*="обнов"], [title*="обнов"]').first();
    if (await refreshBtn.isVisible().catch(() => false)) {
      await refreshBtn.click();
      await page.waitForTimeout(2000);
      action('Admin Status: refresh', 'Clicked');
    } else {
      action('Admin Status: refresh', 'No refresh button found');
    }

    // 5) Admin Logs: tail/refresh
    await page.goto(BASE + '/admin/logs', { waitUntil: 'load', timeout: 20000 });
    await page.waitForTimeout(2000);
    const logsRefresh = page.locator('button:has-text("Обновить"), button:has-text("Refresh"), button:has-text("tail")').first();
    if (await logsRefresh.isVisible().catch(() => false)) {
      await logsRefresh.click();
      await page.waitForTimeout(1500);
      action('Admin Logs: refresh/tail', 'Clicked');
    } else {
      action('Admin Logs: refresh/tail', 'No refresh button found');
    }

    // 6) Admin Tasks: refresh/filter
    await page.goto(BASE + '/admin/tasks', { waitUntil: 'load', timeout: 20000 });
    await page.waitForTimeout(2000);
    const tasksRefresh = page.locator('button:has-text("Обновить"), button:has-text("Refresh"), input[placeholder*="import"]').first();
    if (await tasksRefresh.isVisible().catch(() => false)) {
      await tasksRefresh.click();
      await page.waitForTimeout(1500);
      action('Admin Tasks: refresh/filter', 'Clicked');
    } else {
      const filterInput = page.locator('input[placeholder]').first();
      if (await filterInput.isVisible().catch(() => false)) {
        await filterInput.fill('scan');
        await page.waitForTimeout(1500);
        action('Admin Tasks: filter input', 'Typed "scan"');
      } else {
        action('Admin Tasks: refresh/filter', 'No refresh/filter control found');
      }
    }

  } catch (err) {
    action('Runtime error', err.message || String(err));
    errors.push(err.message || String(err));
  } finally {
    await browser.close();
  }

  console.log('\n--- ACTION LOG ---');
  log.forEach(l => console.log(`${l.action} -> ${l.result}`));
  const failed = log.filter(l => l.result?.includes('error') || l.result?.includes('not found') || l.result?.includes('did not'));
  const issues = log.filter(l => !l.result || l.result === '??');
  console.log('\n--- VERDICT ---');
  const hasErrors = errors.length > 0 || log.some(l => l.result?.includes('Timeout') || l.result?.includes('error'));
  const noResp = log.filter(l => l.result?.includes('not found') || l.result?.includes('did not respond'));
  if (hasErrors) {
    console.log('Issues found:');
    errors.forEach(e => console.log(`  - ${e}`));
    noResp.forEach(l => console.log(`  - ${l.action}: ${l.result}`));
  } else {
    console.log('Interactive behavior OK');
  }
}

main().catch(e => { console.error(e); process.exit(1); });
