import re

# Regular expression to extract PROFILE INFO and total files
profile_pattern = re.compile(r"PROFILE INFO:.*?([\d.]+) ms")
files_pattern = re.compile(r"total files: (\d+)")

data_points = {
    'total_files': [],
    'total_time': [],
    'data_movement_time': [],
    'gpu_function_time': [],
    'metadata_time': [],
    'avg_data_movement_time': [],
    'avg_gpu_function_time': [],
}

# Open and read the posix_output.txt file
with open("gpu_direct_output.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        # Match for total files
        files_match = files_pattern.search(line)
        if files_match:
            total_files = int(files_match.group(1))
            data_points['total_files'].append(total_files)

        # Match for profile information
        match = profile_pattern.search(line)
        if match:
            value = float(match.group(1))
            # Collect data for each of the 6 profile points
            if "total time" in line:
                data_points['total_time'].append(value)
            elif "total data movement time" in line:
                data_points['data_movement_time'].append(value)
            elif "total gpu function execution time" in line:
                data_points['gpu_function_time'].append(value)
            elif "total metadata time" in line:
                data_points['metadata_time'].append(value)
            elif "average data movement per file time" in line:
                data_points['avg_data_movement_time'].append(value)
            elif "average gpu function execution time" in line:
                data_points['avg_gpu_function_time'].append(value)

# Now, 'data_points['total_files']' contains the total number of files
# You can use this list for your X-axis of the plot

# Generate the gnuplot input for each of the 6 plots
plot_data = {
    'total_files': data_points['total_files'],
    'total_time': data_points['total_time'],
    'data_movement_time': data_points['data_movement_time'],
    'gpu_function_time': data_points['gpu_function_time'],
    'metadata_time': data_points['metadata_time'],
    'avg_data_movement_time': data_points['avg_data_movement_time'],
    'avg_gpu_function_time': data_points['avg_gpu_function_time'],
}

# Create a file to store the plot data
with open('plot_data.dat', 'w') as file:
    for i in range(len(data_points['total_files'])):
        file.write(f"{data_points['total_files'][i]} "
                   f"{data_points['total_time'][i]} "
                   f"{data_points['data_movement_time'][i]} "
                   f"{data_points['gpu_function_time'][i]} "
                   f"{data_points['metadata_time'][i]} "
                   f"{data_points['avg_data_movement_time'][i]} "
                   f"{data_points['avg_gpu_function_time'][i]}\n")

# Now you can use gnuplot with plot_data.dat to generate the plots, with total_files as the X-axis
# Here is a gnuplot script to generate the plots:

gnuplot_script = """
set terminal pngcairo enhanced
set output 'profile_plots.png'
set xlabel 'Total Files'
set ylabel 'Time (ms)'
set title 'Profile Information vs Total Files'
plot 'plot_data.dat' using 1:2 title 'Total Time' with lines, \
     'plot_data.dat' using 1:3 title 'Data Movement Time' with lines, \
     'plot_data.dat' using 1:4 title 'GPU Function Time' with lines, \
     'plot_data.dat' using 1:5 title 'Metadata Time' with lines, \
     'plot_data.dat' using 1:6 title 'Avg Data Movement Time' with lines, \
     'plot_data.dat' using 1:7 title 'Avg GPU Function Time' with lines
"""

# Save the gnuplot script to a file
with open("plot_script.gp", "w") as file:
    file.write(gnuplot_script)

print("Data extraction complete. To generate plots, run the following commands in the terminal:")
print("gnuplot plot_script.gp")
