import nox


@nox.session(python=False)
def types(session):
    session.run("pytype", "segmed")


@nox.session(python=False)
def tests(session):
    session.run("pytest")
