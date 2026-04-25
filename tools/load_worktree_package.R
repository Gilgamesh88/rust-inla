load_rustyinla_for_benchmarks <- function(repo_root = getwd()) {
    repo_root <- normalizePath(repo_root, winslash = "/", mustWork = TRUE)
    force_worktree <- identical(Sys.getenv("RUSTYINLA_FORCE_WORKTREE", "0"), "1")

    dll_candidates <- c(
        Sys.getenv("RUSTYINLA_WORKTREE_DLL", ""),
        file.path(
            repo_root,
            "src",
            "rust",
            "target",
            "x86_64-pc-windows-gnu",
            "debug",
            "rustyINLA.dll"
        ),
        file.path(repo_root, "src", "rust", "target", "debug", "rustyINLA.dll"),
        file.path(repo_root, "src", "rustyINLA.dll")
    )
    dll_candidates <- dll_candidates[nzchar(dll_candidates)]
    dll_candidates <- unique(normalizePath(
        dll_candidates[file.exists(dll_candidates)],
        winslash = "/",
        mustWork = FALSE
    ))

    if (!force_worktree && length(dll_candidates) == 0 && requireNamespace("rustyINLA", quietly = TRUE)) {
        suppressPackageStartupMessages(library(rustyINLA, character.only = TRUE))
        return(invisible("installed"))
    }
    if (length(dll_candidates) == 0) {
        stop(
            "Failed to load rustyINLA from the worktree: no development DLL was found.",
            call. = FALSE
        )
    }
    dll_path <- dll_candidates[[1]]
    r_files <- file.path(
        repo_root,
        c("R/extendr-wrappers.R", "R/f.R", "R/interface.R")
    )

    loaded_dlls <- getLoadedDLLs()
    loaded_dll_paths <- vapply(
        loaded_dlls,
        function(info) normalizePath(info[["path"]], winslash = "/", mustWork = FALSE),
        character(1)
    )
    dll_info <- if (dll_path %in% loaded_dll_paths) {
        loaded_dlls[[which(loaded_dll_paths == dll_path)[[1]]]]
    } else {
        dyn.load(dll_path)
    }

    target_env <- globalenv()
    for (r_file in r_files) {
        sys.source(r_file, envir = target_env)
    }
    assign(
        "wrap__rust_inla_run",
        getNativeSymbolInfo("wrap__rust_inla_run", PACKAGE = dll_info),
        envir = target_env
    )

    if (!exists("rusty_inla", envir = target_env, inherits = FALSE)) {
        stop(
            "Failed to load rustyINLA from the worktree: rusty_inla() is still missing.",
            call. = FALSE
        )
    }

    invisible("worktree")
}
