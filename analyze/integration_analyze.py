import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import os, sys

exp_name = 'intgration'
data_path = os.path.join(os.getcwd(), os.pardir, 'results', exp_name)
datafile = os.path.join(data_path, 'Bayesian_multiple_experiments_50.npz')
data = np.load(datafile)
# data = np.load('Bayesian_multiple_experiments_200.npz')
z_decode_gop = data['z_decode_gop']
z_decode_both = data['z_decode_both']
z_decode_mot = data['z_decode_mot']
z_decode_env = data['z_decode_env']

# ground truth
pos_gt = 62
# 将矩阵转换为向量
vector_gop = z_decode_gop.flatten() - pos_gt
vector_both = z_decode_both.flatten() - pos_gt
vector_mot = z_decode_mot.flatten() - pos_gt
vector_env = z_decode_env.flatten() - pos_gt

# hist, bins = np.histogram(vector_gop, bins=10)
hist, bins  = np.histogram(vector_gop, bins=10, density=False, weights=np.ones(len(vector_gop)) / len(vector_gop))
bin_centers = (bins[:-1] + bins[1:]) / 2
mu, std = norm.fit(vector_gop)

# # Plot the Gaussian curve
# plt.plot(x, gaussian_curve, label='Gaussian Fit', color='red')

# def gaussian(x, amplitude, mean, stddev):
#     return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# def normalized_gaussian(x, mean, stddev):
#     return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(vector_gop), np.std(vector_gop)])
# x_fit = np.linspace(min(bin_centers) - 0.05, max(bin_centers) + 0.05, 1000)
# y_fit = gaussian(x_fit, *popt)
# y_fit = normalized_gaussian(x_fit, popt[1], popt[2])

# Generate values for the Gaussian curve
x_fit = np.linspace(min(vector_gop)-0.02, max(vector_gop)+0.02, 200)
y_fit = norm.pdf(x_fit, mu, std) * (bins[1] - bins[0])

save_path = os.path.join(os.getcwd(),os.pardir,'figures')
if not os.path.exists(save_path):
    os.makedirs(save_path)

color = ['#E89DA0','#88CEE6','#B2D3A4', '#F6C8A8']
fontsize = 22
linewidth = 3.
legend_fontsize = 16
scatter_size = 5
plt.figure(figsize=(10, 6))
alpha = 0.8
# density = True
# plt.hist(vector_mot, bins=10, alpha=alpha, label='Motion', color = color[1], density=density)
# plt.hist(vector_env, bins=10, alpha=alpha, label='Environment', color = color[3], density=density)
# plt.hist(vector_both, bins=10, alpha=alpha, label='Integration', color = color[0], density=density)
# plt.plot(x_fit, y_fit, label='Theo. Bayes.', linewidth=5, color = color[2])
plt.hist(vector_mot, bins=10, alpha=alpha, label='Motion', color=color[1], density=False, weights=np.ones(len(vector_mot)) / len(vector_mot))
plt.hist(vector_env, bins=10, alpha=alpha, label='Environment', color=color[3], density=False, weights=np.ones(len(vector_env)) / len(vector_env))
plt.hist(vector_both, bins=10, alpha=alpha, label='Integration', color=color[0], density=False, weights=np.ones(len(vector_both)) / len(vector_both))
plt.plot(x_fit, y_fit, label='Theo. Bayes.', linewidth=linewidth, color=color[2])
plt.xlim([-0.15, 0.15])
plt.xticks([-0.1, 0, 0.1], ['-0.1', '0', '0.1'], fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.legend(fontsize=legend_fontsize)
plt.xlabel('Estimated Position', fontsize=fontsize)
plt.ylabel('Probability', fontsize=fontsize)
plt.yticks([0, 0.1, 0.2, 0.3], ['0', '0.1', '0.2', '0.3'], fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'integration_hist.png'))
# plt.show()

import scipy.stats as stats
# 1,2,3,4: only motion, only environment, both, bayesian(gop)
# calculate the squared error for each 
# se_motion = vector_mot**2
# se_env = vector_env**2
# se_both = vector_both**2
# se_gop = vector_gop**2
se_motion = np.abs(vector_mot)
se_env = np.abs(vector_env)
se_both = np.abs(vector_both)
se_gop = np.abs(vector_gop)

# filter the outliers by quartiles
def filter_outliers(data):
    q1 = np.percentile(data, 30)
    q3 = np.percentile(data, 70)
    iqr = q3 - q1
    lower_bound = q1 - 1.25 * iqr
    upper_bound = q3 + 1.25 * iqr
    return data[(data > lower_bound) & (data < upper_bound)]

se_motion = filter_outliers(se_motion)
se_env = filter_outliers(se_env)
se_both = filter_outliers(se_both)
se_gop = filter_outliers(se_gop)

# Perform independent t-tests
ttest_1_2 = stats.ttest_ind(se_motion, se_env, equal_var=False)
ttest_1_3 = stats.ttest_ind(se_motion, se_both, equal_var=False)
ttest_2_3 = stats.ttest_ind(se_env, se_both, equal_var=False)
ttest_3_4 = stats.ttest_ind(se_motion, se_gop, equal_var=False)

print('Motion vs. Environment:', ttest_1_2)
print('Motion vs. Both:', ttest_1_3)
print('Environment vs. Both:', ttest_2_3)
print('Both vs. Theo. Bayes.:', ttest_3_4)


# Create bar plot with error bars
means = [np.mean(se_motion), np.mean(se_env), np.mean(se_both), np.mean(se_gop)]
sems = [stats.sem(se_motion), stats.sem(se_env), stats.sem(se_both), stats.sem(se_gop)]

labels = ['1', '2', '3', '4']
def add_significance_text(ax, x1, x2, y, h, p_value, offset_index):
    """
    Add a line and significance label ('ns' for not significant, '*' for significant) 
    between two bars at staggered heights.
    
    Parameters:
    ax: The axis object to add the annotation to.
    x1, x2: The x positions of the bars to compare.
    y: The y position of the line end.
    h: The height of the lines connecting the bars.
    p_value: The p-value used to determine if a star should be added.
    offset_index: The index to determine the height offset for staggered lines.
    """
    # Stagger the heights of the lines
    stagger = offset_index * 1e-2 * 1.5
    ax.plot([x1, x1, x2, x2], [y+h+stagger, y+stagger, y+stagger, y+h+stagger], lw=0.4, c='k')
    # Based on the p-value, choose the label
    label = '*' if p_value < 0.05 else 'ns'
    if p_value >= 0.05:
        label = 'ns'
    elif p_value<0.05 and p_value>=0.01:
        label = '*'
    elif p_value<0.01 and p_value>=0.001:
        label = '**'
    else:
        label = '***'
    # Output to the fourth decimal place
    # p_value = round(p_value, 4)
    # label = 'p = ' + str(p_value)
    ax.text((x1+x2)*0.5, y+stagger, label, ha='center', va='bottom', color='k')

# Create a new plot with staggered significance lines
fig, ax = plt.subplots(figsize=(8, 5))

# Bar plot
bars = ax.bar(labels, means, yerr=sems, capsize=10, color=color)

# Calculate the y position for the significance line, which will be above the error bars
y_offsets = [sem  for sem in sems]  # This will be the distance above the error bars
y_values = [mean + offset for mean, offset in zip(means, y_offsets)]

# The maximum y value for drawing the significance lines
max_y = max(y_values)

# Add staggered significance lines
# The offset_index parameter will stagger the heights of the significance lines
h = 0.002
add_significance_text(ax, 0, 1, max_y, h, ttest_1_2.pvalue, 0.2)
add_significance_text(ax, 0, 2, max_y, h, ttest_1_3.pvalue, 0.4)
add_significance_text(ax, 1, 2, max_y, h, ttest_2_3.pvalue, 0.6)
add_significance_text(ax, 2, 3, max_y, h, ttest_3_4.pvalue, 0.8)

# ax.set_ylabel('Squared error', fontsize=fontsize)
ax.set_ylabel('Decoded error', fontsize=fontsize)
# y ticks [0,0.01,0.02,0.03]
ax.set_yticks([0,0.01,0.02,0.03],['0','0.01','0.02','0.03'], fontsize=fontsize)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(['Motion', 'Environment', 'Both', 'Theo. Bayes.'], fontsize=fontsize)
plt.tight_layout()

plt.savefig(os.path.join(save_path, 'integration_bar.png'))
# plt.show()

