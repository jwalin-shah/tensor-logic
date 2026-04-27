domain Person { jwalin ryan steven }
domain Topic { tensor_logic robotics local_ai openhuman }
domain Project { openhuman tensor_demo jarvis }

relation mentioned(Person, Topic)
relation project_about(Project, Topic)
relation should_follow_up(Person, Project)

fact mentioned(ryan, tensor_logic)
fact mentioned(ryan, robotics)
fact mentioned(steven, local_ai)

fact project_about(tensor_demo, tensor_logic)
fact project_about(openhuman, local_ai)
fact project_about(jarvis, robotics)

rule should_follow_up(p,proj) := (mentioned(p,t) * project_about(proj,t)).step()

query should_follow_up(ryan, tensor_demo)
prove should_follow_up(ryan, tensor_demo)
