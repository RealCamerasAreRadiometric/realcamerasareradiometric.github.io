# docs_tests.jl — the docstring gates the Documenter build cannot run, through VisualGeometryToolkitCore's `test_docs` (its TestExt). Its `checkdocs = :public` only sees docstrings that exist, so
# a public name with none, a docstring detached from its definition by a blank or comment line (Julia 1.12 makes it a bare string)
# and a signature line that drifted from the methods (an argument or keyword no method takes) all satisfy it.
#
# Two lists, kept apart:
# - EXCEPTIONS: a name that is not documented ON PURPOSE, with its reason.
# - BACKLOG: names still to document (the docs review of 2026-11-04 stopped at these). The list only shrinks: a name documented must
#   leave it, and a new public name without a docstring fails.

@testitem "docs: every public name has a docstring, attached, and its signatures match the methods" begin
    using Test, RadiometricCamera
    import VisualGeometryToolkitCore as VgtkCore
    VgtkCore.test_docs(RadiometricCamera)
end
