const sourceEl = document.getElementById("source");
const resultEl = document.getElementById("result");
const relationEl = document.getElementById("relation");
const arg1El = document.getElementById("arg1");
const arg2El = document.getElementById("arg2");
const recursiveEl = document.getElementById("recursive");

sourceEl.value = `domain Person { alice bob cara }

relation parent(Person, Person)
relation ancestor(Person, Person)

fact parent(alice, bob)
fact parent(bob, cara)

rule ancestor(x,z) := (parent(x,z) + ancestor(x,y) * parent(y,z)).step()

query ancestor(alice, cara) recursive
prove ancestor(alice, cara) recursive`;

function payload() {
  return {
    source: sourceEl.value,
    relation: relationEl.value,
    arg1: arg1El.value,
    arg2: arg2El.value,
    recursive: recursiveEl.checked,
  };
}

async function invoke(action) {
  resultEl.textContent = `$ ${action}...`;
  try {
    const res = await fetch(`/api/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    const data = await res.json();
    if (!res.ok) {
      resultEl.textContent = `HTTP ${res.status}\n${JSON.stringify(data, null, 2)}`;
      return;
    }
    resultEl.textContent = `${data.stdout || ""}${data.stderr ? `\n[stderr]\n${data.stderr}` : ""}`.trim();
  } catch (error) {
    resultEl.textContent = String(error);
  }
}

document.querySelectorAll("button[data-action]").forEach((button) => {
  button.addEventListener("click", () => invoke(button.dataset.action));
});
