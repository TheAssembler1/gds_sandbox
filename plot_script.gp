
set terminal pngcairo enhanced
set output 'profile_plots.png'
set xlabel 'Total Files'
set ylabel 'Time (ms)'
set title 'Profile Information vs Total Files'
plot 'plot_data.dat' using 1:2 title 'Total Time' with lines,      'plot_data.dat' using 1:3 title 'Data Movement Time' with lines,      'plot_data.dat' using 1:4 title 'GPU Function Time' with lines,      'plot_data.dat' using 1:5 title 'Metadata Time' with lines,      'plot_data.dat' using 1:6 title 'Avg Data Movement Time' with lines,      'plot_data.dat' using 1:7 title 'Avg GPU Function Time' with lines
