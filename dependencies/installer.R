for (package in readLines('requirements.R.txt')) {
    p <- base::strsplit(package, ",")
    name <- p[[1]][1]
    version <- p[[1]][2]
    if (!name %in% rownames(installed.packages())) {
        remotes::install_version(name, version = version, repos = "http://cran.us.r-project.org")
    }
}

