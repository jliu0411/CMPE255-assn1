# Momentum Tasks

A polished, dependency-free full-stack task workspace with persistent storage, responsive design, dark mode, keyboard shortcuts, search, smart sorting, priorities, tags, dates, and status workflows.

## Run

`fullstack-test` is the project directory, not a file to execute. Requires Node.js 20 or newer.

From the assignment workspace root (`CMPE255-assignment`):

```powershell
cd ".\CMPE255-assn1\Part2\00_dynamic_todo_workspace\fullstack-test"
npm start
```

If your terminal is already at the `CMPE255-assn1` repository root:

```powershell
cd ".\Part2\00_dynamic_todo_workspace\fullstack-test"
npm start
```

If your prompt already ends with `Part2\00_dynamic_todo_workspace>` (as in the screenshot):

```powershell
cd ".\fullstack-test"
npm start
```

If your terminal is already inside `fullstack-test`, run only:

```powershell
npm start
```

Open `http://localhost:4173`. Data is created automatically in `data/tasks.json`.

For watch mode use `npm run dev`. Run the tests with `npm test`.

## Shortcuts

- `N` — create a task
- `/` — focus search
- `Esc` — close the mobile navigation
