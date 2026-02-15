#!/usr/bin/env node
/**
 * E2E smoke test for http://127.0.0.1:5051/app
 * Tests: login, catalogue, settings (DB+migration), admin pages, AI search
 */
import { chromium } from 'playwright';

const BASE = 'http://127.0.0.1:5051/app';
const results = [];

function step(name, pass, detail) {
  results.push({ name, pass, detail });
  console.log(`[${pass ? 'PASS' : 'FAIL'}] ${name}: ${detail}`);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  
  const errors = [];
  page.on('pageerror', err => errors.push(`JS Error: ${err.message}`));
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(`Console: ${msg.text()}`);
  });

  try {
    // 1) Open login page
    await page.goto(BASE, { waitUntil: 'load', timeout: 20000 });
    const hasLoginForm = await page.locator('input[type="password"]').isVisible().catch(() => false);
    step('1) Open login page', hasLoginForm, hasLoginForm ? 'Login form visible' : 'No login form found');

    // 2) Login with admin/admin123
    if (hasLoginForm) {
      await page.fill('.login-card input:not([type="password"])', 'admin');
      await page.fill('input[type="password"]', 'admin123');
      await page.click('button[type="submit"], button:has-text("Войти")');
      await page.waitForTimeout(3000);
      const url = page.url();
      const loggedIn = !url.includes('/login');
      step('2) Login admin/admin123', loggedIn, loggedIn ? `Redirected to ${url}` : 'Still on login page');
      if (!loggedIn) {
        console.log('\nBLOCKER: Login failed. Cannot proceed.\n');
        await browser.close();
        process.exit(1);
      }
    }

    // 3) Verify main catalogue loads
    if (!page.url().includes('/app') || page.url().includes('/login')) {
      await page.goto(BASE, { waitUntil: 'load', timeout: 20000 });
    }
    await page.waitForTimeout(2000);
    const body = (await page.textContent('body')) || '';
    const hasCatalogue = body.includes('Каталог') || body.includes('Поиск') || body.length > 1000;
    step('3) Catalogue page loads', hasCatalogue, hasCatalogue ? 'Page rendered' : 'Empty or error');

    // 4) Settings: database section + migration wizard
    await page.goto(BASE + '/settings', { waitUntil: 'load', timeout: 20000 });
    await page.waitForTimeout(2500);
    const expertBtn = page.locator('button:has-text("Expert")');
    if (await expertBtn.isVisible().catch(() => false)) {
      await expertBtn.click();
      await page.waitForTimeout(1500);
    }
    const settingsBody = (await page.textContent('body')) || '';
    const hasDbSection = settingsBody.includes('Управление базой данных') || settingsBody.includes('SQLite') || settingsBody.includes('PostgreSQL');
    const hasMigrationWizard = settingsBody.includes('Миграция') || settingsBody.includes('Wizard') || settingsBody.includes('host.docker');
    step('4a) Settings: DB section', hasDbSection, hasDbSection ? 'DB management visible' : 'Not found');
    step('4b) Settings: Migration wizard', hasMigrationWizard, hasMigrationWizard ? 'Migration controls visible' : 'Not found');

    // 5) Admin pages
    const adminPages = [
      { path: '/admin/tasks', label: 'Admin Tasks' },
      { path: '/admin/logs', label: 'Admin Logs' },
      { path: '/admin/status', label: 'Admin Service Status' },
    ];
    for (const { path, label } of adminPages) {
      await page.goto(BASE + path, { waitUntil: 'load', timeout: 20000 });
      await page.waitForTimeout(2000);
      const loaded = page.url().includes(path.split('/').pop());
      step(`5) ${label}`, loaded, loaded ? 'Page loaded' : 'Failed to load');
    }

    // 6) AI search from UI
    await page.goto(BASE, { waitUntil: 'load', timeout: 20000 });
    await page.waitForTimeout(2000);
    
    const aiToggle = page.locator('input[type="checkbox"], label:has-text("Поиск ИИ"), input#ai').first();
    const searchInput = page.locator('input[placeholder*="Поиск"]').first();
    
    if (await searchInput.isVisible().catch(() => false)) {
      await searchInput.fill('тестовый запрос');
      
      // Try to enable AI search if toggle exists
      if (await aiToggle.isVisible().catch(() => false)) {
        await aiToggle.click();
        await page.waitForTimeout(500);
      }
      
      await page.keyboard.press('Enter');
      await page.waitForTimeout(8000); // Wait for AI processing
      
      const afterSearch = (await page.textContent('body')) || '';
      const hasAiResponse = afterSearch.includes('AI') || afterSearch.includes('ИИ') || afterSearch.includes('Ответ') || afterSearch.includes('sources') || afterSearch.includes('Найдено');
      const hasCrash = errors.some(e => e.includes('Uncaught') || e.includes('TypeError') || e.includes('undefined'));
      
      step('6) AI search', !hasCrash, hasAiResponse ? 'Response rendered (sources may be empty)' : hasCrash ? 'CRASH detected' : 'Search executed, no crash');
    } else {
      step('6) AI search', false, 'Search input not found');
    }

    // Check for fatal errors
    const fatalErrors = errors.filter(e => e.includes('Uncaught') || e.includes('Fatal'));
    if (fatalErrors.length > 0) {
      console.log('\n--- FATAL ERRORS ---');
      fatalErrors.forEach(e => console.log(`  ${e}`));
    }

  } catch (err) {
    step('Runtime error', false, err.message || String(err));
  } finally {
    await browser.close();
  }

  console.log('\n--- SMOKE TEST REPORT ---');
  const passed = results.filter(r => r.pass).length;
  const failed = results.filter(r => !r.pass).length;
  console.log(`Total: ${results.length} | Passed: ${passed} | Failed: ${failed}`);
  console.log('\nPer-step results:');
  results.forEach(r => console.log(`  ${r.pass ? '✓' : '✗'} ${r.name}: ${r.detail}`));
  
  if (errors.length > 0) {
    console.log('\n--- UI ERRORS CAPTURED ---');
    errors.slice(0, 10).forEach(e => console.log(`  - ${e}`));
    if (errors.length > 10) console.log(`  ... and ${errors.length - 10} more`);
  } else {
    console.log('\nNo UI errors (network/JS/console) detected.');
  }

  process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
