
set terminal pngcairo size 1920,1080 enhanced
set output "profile_plots.png"
set multiplot layout 2,3 title "Profile Info Plots"
set xlabel "Number of Files"
set ylabel "Time (ms)"

# Plot total time
plot "plot_data.dat" using 1:2 with lines title "Total Time"

# Plot data movement time
plot "plot_data.dat" using 1:3 with lines title "Data Movement Time"

# Plot GPU function time
plot "plot_data.dat" using 1:4 with lines title "GPU Function Time"

# Plot metadata time
plot "plot_data.dat" using 1:5 with lines title "Metadata Time"

# Plot avg data movement per file time
plot "plot_data.dat" using 1:6 with lines title "Avg Data Movement Time"

# Plot avg gpu function execution time
plot "plot_data.dat" using 1:7 with lines title "Avg GPU Function Time"
