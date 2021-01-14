using CSV
using DrWatson
using DataFrames
using Plots

data_dir = datadir("exp_raw/TestCLIM_N-5_T-100")

data_str = data_dir * "/TestCLIM_N-5_T-100_0191.txt"

frames = DataFrame.(CSV.File.(readdir(data_dir, join = true), header = 0))

big_df = hcat(frames..., makeunique=true)

write_path = datadir("exp_pro/TestCLIM_N-5_T-100.csv")

CSV.write(write_path, big_df, writeheader = false)
