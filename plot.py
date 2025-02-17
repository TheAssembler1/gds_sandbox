import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('profile_output.csv', header=None, names=['operation', 'read', 'file_num', 'stat', 'value'])

# List of unique stats (4th column)
unique_stats = df['stat'].unique()

# Create a plot for each unique stat
for stat in unique_stats:
    # Filter the data for the current stat
    stat_data = df[df['stat'] == stat]
    
    # Remove rows where value is 0
    stat_data = stat_data[stat_data['value'] != 0]
    
    # Get the unique operations in the first column for grouping
    operations = stat_data['operation'].unique()

    # Create a new figure for each unique stat
    plt.figure(figsize=(10, 6))

    # Plot the data for each operation
    for operation in operations:
        # Filter the data for the current operation
        operation_data = stat_data[stat_data['operation'] == operation]
        
        # Plot x as 'file_num', y as 'value', and label as operation
        plt.plot(operation_data['file_num'], operation_data['value'], label=operation)
    
    # Set the title, labels, and grid
    plt.title(stat)
    plt.xlabel('Number of Files')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f"{stat}_plot.png")

    # Show the plot
    plt.show()
