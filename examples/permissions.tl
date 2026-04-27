domain User { alice bob }
domain Role { admin viewer }
domain Resource { invoice dashboard customer_record }

relation has_role(User, Role)
relation role_can_access(Role, Resource)
relation can_access(User, Resource)

fact has_role(alice, admin)
fact has_role(bob, viewer)

fact role_can_access(admin, invoice)
fact role_can_access(admin, customer_record)
fact role_can_access(viewer, dashboard)

rule can_access(u,r) := (has_role(u,role) * role_can_access(role,r)).step()

query can_access(alice, customer_record)
prove can_access(alice, customer_record)
