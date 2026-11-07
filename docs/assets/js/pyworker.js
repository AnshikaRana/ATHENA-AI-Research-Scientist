/* Athena - Python analysis sandbox (browser port of run_python_analysis).
 *
 * Runs in a Web Worker with Pyodide. The same static AST policy as the Python
 * version is applied before execution, and the main thread terminates this
 * worker after 15 seconds - the browser equivalent of the short-lived
 * subprocess with a hard timeout.
 */
let pyodide = null;

const RUNNER = `
import ast, io, json, sys, contextlib, textwrap
ALLOWED_IMPORTS = {"math","statistics","random","json","itertools","collections",
                   "decimal","fractions","datetime","re","numpy"}
BANNED_NAMES = {"open","exec","eval","compile","__import__","input","breakpoint",
                "globals","locals","vars","getattr","setattr","delattr","memoryview"}

def _validate(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{alias.name}' is not permitted. Allowed: {sorted(ALLOWED_IMPORTS)}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not permitted. Allowed: {sorted(ALLOWED_IMPORTS)}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of '{node.id}' is not permitted"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"access to dunder attribute '{node.attr}' is not permitted"
    return None

def _run(code):
    code = textwrap.dedent(code)
    problem = _validate(code)
    if problem:
        return json.dumps({"unsafe": problem})
    out, err = io.StringIO(), io.StringIO()
    status = 0
    sys.setrecursionlimit(2000)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, "<analysis>", "exec"), {"__name__": "__main__"})
        except SystemExit as exc:
            status = exc.code if isinstance(exc.code, int) else 1
        except BaseException:
            import traceback
            traceback.print_exc()
            status = 1
    return json.dumps({"status": status, "stdout": out.getvalue(), "stderr": err.getvalue()})

_run(__athena_code)
`;

self.onmessage = async (event) => {
  const { code, indexURL } = event.data || {};
  try {
    if (!pyodide) {
      importScripts(indexURL + "pyodide.js");
      // eslint-disable-next-line no-undef
      pyodide = await loadPyodide({ indexURL });
    }
    if (/\bnumpy\b/.test(code)) await pyodide.loadPackage("numpy");
    self.postMessage({ type: "started" });
    pyodide.globals.set("__athena_code", code);
    const raw = pyodide.runPython(RUNNER);
    self.postMessage({ type: "result", payload: JSON.parse(raw) });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
