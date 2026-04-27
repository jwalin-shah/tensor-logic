domain Task {
  design
  backend
  frontend
  tests
  launch
}

relation blocks(Task, Task)
relation must_finish_before(Task, Task)

fact blocks(design, backend)
fact blocks(backend, frontend)
fact blocks(frontend, tests)
fact blocks(tests, launch)

rule must_finish_before(x,z) := (blocks(x,z) + must_finish_before(x,y) * blocks(y,z)).step()

query must_finish_before(design, launch) recursive
prove must_finish_before(design, launch) recursive
