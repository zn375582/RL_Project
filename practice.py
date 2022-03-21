import matplotlib.pyplot as plt

num_runs_options = [1, 100, 1000, 10000]
pi_time = [0.5, 1, 2.5, ]
vi_time = [0.1, .3, 1, ]


# Create graph to compare time execution of both iterations
fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.
ax.plot(num_runs_options, vi_time, label="Value iteration:")
ax.plot(num_runs_options, pi_time, label="Policy iteration:")
ax.grid(True)
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Time taken')  # Add a y-label to the axes.
ax.set_title("Execution Time: Value Iteration vs Policy Iteration")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.show()