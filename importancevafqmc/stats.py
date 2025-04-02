import pstats

# Load the profiling data from the output file
p = pstats.Stats('profiler_output.prof')

# Sort the output by cumulative time and print the top 50 entries
print("hi")
p.sort_stats('cumulative').print_stats(50)