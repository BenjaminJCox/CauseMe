using CSV
using DrWatson
using DataFrames
using Plots

which_data = "TestCLIMnoise_N-5_T-100"

data_dir = datadir("exp_raw/" * which_data)


frames = DataFrame.(CSV.File.(readdir(data_dir, join = true), header = 0))

big_df = hcat(frames..., makeunique=true)

write_path = datadir("exp_pro/" * which_data * ".csv")

CSV.write(write_path, big_df, header = false)
