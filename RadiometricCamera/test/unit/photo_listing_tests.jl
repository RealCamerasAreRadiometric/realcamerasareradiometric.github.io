# The photographs of a directory (owner, 2026-11-15): RAW and PNG files by their extension, never a macOS AppleDouble sidecar
# (`._<name>`, the resource fork a copy from a Mac leaves beside every file) nor Finder's `.DS_Store` — both were taken for photographs
# before. ONE predicate per kind (`is_raw_file`, `is_png_file`) and one listing (`photo_files`); every RC and experiments listing uses them.

@testitem "photo listing: RAW and PNG files, never ._* or .DS_Store" begin
    using Test, RadiometricCamera
    const RC = RadiometricCamera
    d = mktempdir()
    for f in ("DSC1.ARW", "._DSC1.ARW", "V01.png", "._V01.png", ".DS_Store", "notes.txt", "DSC2.dng")
        write(joinpath(d, f), "x")
    end
    @test RC.photo_files(d) == ["DSC1.ARW", "DSC2.dng", "V01.png"]
    @test RC.photo_files(d; join = true) == joinpath.(d, ["DSC1.ARW", "DSC2.dng", "V01.png"])
    @test RC.is_raw_file("DSC1.ARW") && !RC.is_raw_file("._DSC1.ARW") && !RC.is_raw_file(joinpath(d, "._DSC1.ARW"))
    @test RC.is_png_file("V01.png") && !RC.is_png_file("._V01.png") && !RC.is_png_file(".DS_Store")
    @test RC.raw_file(d, "DSC1.png") == joinpath(d, "DSC1.ARW")          # a frame's RAW file: never its sidecar
end
