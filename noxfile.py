import nox


@nox.session(python=False)
def types(session):
    session.run("pytype", "segmed")


@nox.session(python=False)
def tests(session):
    session.run("pytest")


@nox.session(python=False)
def docs(session):
    # build the source code docs
    session.run("sphinx-apidoc", "-f", "-o", "docs/source", "segmed/")
    # build all the documentation to html
    session.run("sphinx-build", "-b", "html", "docs/source", "docs/build")
