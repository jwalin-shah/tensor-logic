domain Module {
  api
  db
  models
  worker
  auth
}

relation imports(Module, Module)
relation depends_on(Module, Module)

fact imports(worker, api)
fact imports(api, db)
fact imports(db, models)
fact imports(api, auth)

rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()

query depends_on(worker, models) recursive
prove depends_on(worker, models) recursive
