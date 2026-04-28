const sourceEl = document.getElementById("source");
const resultEl = document.getElementById("result");
const relationEl = document.getElementById("relation");
const arg1El = document.getElementById("arg1");
const arg2El = document.getElementById("arg2");
const recursiveEl = document.getElementById("recursive");

sourceEl.value = `# Minimal sample
parent(alice,bob).
parent(bob,cara).

ancestor(X,Y) :- parent(X,Y).
ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).

query ancestor(alice,cara).`;

function payload() {
  return {
    source: sourceEl.value,
    relation: relationEl.value,
    arg1: arg1El.value,
    arg2: arg2El.value,
    recursive: recursiveEl.checked,
  };
}

function mockResponse(action, body) {
  if (action === "query") {
    return {
      action,
      mock: true,
      stdout: `${body.relation}(${body.arg1}, ${body.arg2}) = <mock result>`,
      stderr: "",
      exit_code: 0,
    };
  }
  return {
    action,
    mock: true,
    stdout: `Mock ${action} completed. Connect /api/${action} to backend to execute real tensor_logic commands.`,
    stderr: "",
    exit_code: 0,
  };
}

async function invoke(action) {
  const body = payload();
  resultEl.textContent = `$ ${action}...`;

  try {
    const res = await fetch(`/api/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      resultEl.textContent = `HTTP ${res.status}\n${JSON.stringify(data, null, 2)}`;
      return;
    }
    resultEl.textContent = `${data.stdout || ""}${data.stderr ? `\n[stderr]\n${data.stderr}` : ""}`.trim();
  } catch {
    const data = mockResponse(action, body);
    resultEl.textContent = `[mock mode]\n${data.stdout}`;
  }
}

document.querySelectorAll("button[data-action]").forEach((button) => {
  button.addEventListener("click", () => invoke(button.dataset.action));
});
