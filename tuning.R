lib_path = "./rlibs"
if (!require("irace", lib.loc = lib_path, quietly = TRUE)) {
  install.packages("irace", lib = lib_path, repos = "http://cran.us.r-project.org")
  library("irace", lib.loc = lib_path)
}

setwd("./tuning")
scenario <- readScenario(filename = "scenario.txt", scenario = defaultScenario())
irace.main(scenario = scenario)
