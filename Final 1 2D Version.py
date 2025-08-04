import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy import stats
# Parameters
J = 8  
N_walks = 20000
max_steps = 100
max_steps_2d = 200
a = 1.0  
h_bar = 1.0  
m = 1.0  
trim_index = J 
trim_end_1d = 15  
trim_end_2d = 50

def random_walk_1d(J, max_steps, excited=False):
    position = 0
    for step in range(max_steps):
        move = np.random.choice([-1, 1])  
        position += move

        # Ground state: Only arrest at boundary
        if position == -J or position == J:
            return step
        
        # First excited state: Also arrest if crossing zero point but after a step to not insta-arrest
        if excited and position == 0 and step > 0:
            return step  
    
    return max_steps

def linear_func(x, a, b):
    return a * x + b


def calc_arrest_rate(survival_times):
    step_counts = np.arange(1, max_steps + 1)
    survival_counts = [np.sum(np.array(survival_times) >= n) for n in step_counts]
    log_survival_counts = np.log(np.maximum(survival_counts, 1e-10))
    
    trimmed_steps = step_counts[trim_index:-trim_end_1d]
    trimmed_log_survival_counts = log_survival_counts[trim_index:-trim_end_1d]
    
    
    survival_counts_array = np.array(survival_counts[trim_index:-trim_end_1d])
    weights = survival_counts_array / np.sum(survival_counts_array)
    avg_relative_error = np.mean(1.0/np.sqrt(survival_counts_array))
    log_survival_errors = np.where(survival_counts_array > 0, 
                                 avg_relative_error * np.sqrt(1 + weights),
                                 0)
    
    
    popt, pcov = curve_fit(linear_func, trimmed_steps, trimmed_log_survival_counts,
                          sigma=log_survival_errors, absolute_sigma=True)
                      
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))
    
    
    predicted = linear_func(trimmed_steps, slope, intercept)
    residuals = trimmed_log_survival_counts - predicted
    chisq = np.sum((residuals/log_survival_errors)**2)
    reduced_chisq = chisq / (len(trimmed_steps) - 2)
    
    lambda_estimate = -slope
    lambda_err = slope_err
    
    return (lambda_estimate, lambda_err, trimmed_steps, trimmed_log_survival_counts,
            slope, intercept, slope_err, intercept_err, reduced_chisq, log_survival_errors)

def plot_residual_analysis(steps, log_counts, slope, intercept, title_prefix):
    
    predicted = linear_func(steps, slope, intercept)
    residuals = log_counts - predicted
    std_dev = np.std(residuals)
    
    
    standardized_residuals = (residuals - np.mean(residuals)) / std_dev
    D, p_value = stats.kstest(standardized_residuals, 'norm')
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix} Residual Analysis')
    
    # 1. Residuals vs Steps plot
    ax1.scatter(steps, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Steps')
    
    # 2. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    # 3. Histogram with normal distribution
    n_bins = 30
    counts, bins, _ = ax3.hist(residuals, bins=n_bins, density=True, alpha=0.7)
    x = np.linspace(min(bins), max(bins), 100)
    ax3.plot(x, stats.norm.pdf(x, np.mean(residuals), std_dev), 
             'r-', lw=2, label=f'Normal\nσ={std_dev:.3f}')
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Histogram of Residuals')
    ax3.legend()
    
    # 4. Statistics text box (now includes KS test results)
    stats_text = f"""Standard Deviation: {std_dev:.4f}
    KS Test:
    D-statistic: {D:.4f}
    p-value: {p_value:.4f}"""
    
    ax4.text(0.5, 0.5, stats_text, 
             horizontalalignment='center', verticalalignment='center',
             transform=ax4.transAxes, fontsize=12)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
# 1D Calculations
survival_times_1d = [random_walk_1d(J, max_steps) for _ in range(N_walks)]
survival_times_1d_excited = [random_walk_1d(J, max_steps, excited=True) for _ in range(N_walks)]
# Ground State
(lambda_ground_1d, lambda_ground_err_1d, step_counts, log_survival_counts,
 slope, intercept, slope_err, intercept_err, red_chi2_ground, log_errors) = calc_arrest_rate(survival_times_1d)
# Excited State
(lambda_excited_1d, lambda_excited_err_1d, steps_excited, log_counts_excited,
 slope_excited, intercept_excited, slope_err_excited, intercept_err_excited,
 red_chi2_excited, log_errors_excited) = calc_arrest_rate(survival_times_1d_excited)

# Error propagation
E_ground_1d = (h_bar**2 / (m * a**2)) * (lambda_ground_1d * J**2)
E_ground_err_1d = (h_bar**2 / (m * a**2)) * (lambda_ground_err_1d * J**2)
E_excited_1d = (h_bar**2 / (m * a**2)) * (lambda_excited_1d * J**2)
E_excited_err_1d = (h_bar**2 / (m * a**2)) * (lambda_excited_err_1d * J**2)

print("\n1D Results:")
print(f"Ground State Energy: {E_ground_1d:.4f} ± {E_ground_err_1d:.4f}")
print(f"Excited State Energy: {E_excited_1d:.4f} ± {E_excited_err_1d:.4f}")
print(f"Energy Ratio (1D): {E_excited_1d/E_ground_1d:.2f}")
print(f"Ground State Reduced χ²: {red_chi2_ground:.2f}")
print(f"Excited State Reduced χ²: {red_chi2_excited:.2f}")

plt.figure(figsize=(12,8))
plt.errorbar(step_counts, log_survival_counts, yerr=log_errors, 
             fmt='o', markersize=3, label="Ground State Data", color="blue")
plt.plot(step_counts, slope * step_counts + intercept, 
         linestyle="dashed", color="blue", label="Ground State Fit")
plt.xlabel("Number of Steps")
plt.ylabel("Log(Survival Counts)")
plt.title("1D Ground State Random Walk")
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.errorbar(steps_excited, log_counts_excited, yerr=log_errors_excited,
             fmt='o', markersize=3, label="Excited State Data", color="red")
plt.plot(steps_excited, slope_excited * steps_excited + intercept_excited, 
         linestyle="dashed", color="red", label="Excited State Fit")
plt.xlabel("Number of Steps")
plt.ylabel("Log(Survival Counts)")
plt.title("1D Excited State Random Walk")
plt.legend()
plt.show()

########################################################
################ 2 DIMENSIONAL VERSION #################
########################################################
trim_index_2d = 2*J

def random_walk_2d(J, max_steps_2d, excited=False):
    
    if excited:
        j_1 = 2
        j_2 = np.random.randint(-J//2, J//2)
    else:
        j_1, j_2 = 0, 0
    
    prev_j1 = j_1
    
    for step in range(max_steps_2d):
        direction = np.random.choice(["x+", "x-", "y+", "y-"])
        new_j1, new_j2 = j_1, j_2
        
        if direction == "x+":
            new_j1 += 1
        elif direction == "x-":
            new_j1 -= 1
        elif direction == "y+":
            new_j2 += 1
        elif direction == "y-":
            new_j2 -= 1
        
        # Check if the new position would be outside the boundary
        if new_j1**2 + new_j2**2 >= J**2:
            return step
        
        j_1, j_2 = new_j1, new_j2
        
        if excited and step > 3:
            if (prev_j1 > 0 and j_1 <= 0) or (prev_j1 < 0 and j_1 >= 0):
                return step
        
        prev_j1 = j_1
    
    return max_steps_2d



def calc_arrest_rate_2d(survival_times):
    step_counts_2d = np.arange(1, max_steps_2d + 1)
    survival_counts_2d = [np.sum(np.array(survival_times) >= n) for n in step_counts_2d]
    
    min_count = 1.0  # minimum count to avoid log(0)
    log_survival_counts_2d = np.log(np.maximum(survival_counts_2d, min_count))
    
    trimmed_steps_2d = step_counts_2d[trim_index_2d:-trim_end_2d]
    trimmed_log_survival_counts_2d = log_survival_counts_2d[trim_index_2d:-trim_end_2d]
    
    
    survival_counts_array_2d = np.array(survival_counts_2d[trim_index_2d:-trim_end_2d])
    
    # Only use non-zero counts for weight calculation
    nonzero_counts = survival_counts_array_2d[survival_counts_array_2d > 0]
    if len(nonzero_counts) > 0:
        weights_2d = survival_counts_array_2d / np.sum(nonzero_counts)
        avg_relative_error_2d = np.mean(1.0/np.sqrt(nonzero_counts))
    else:
        weights_2d = np.ones_like(survival_counts_array_2d)
        avg_relative_error_2d = 1.0
    
    log_survival_errors_2d = np.where(survival_counts_array_2d > 0,
                                    avg_relative_error_2d * np.sqrt(1 + weights_2d),
                                    1.0)
    
    
    p0 = [-0.01, np.max(trimmed_log_survival_counts_2d)]
    popt_2d, pcov_2d = curve_fit(linear_func, trimmed_steps_2d, trimmed_log_survival_counts_2d,
                                p0=p0, sigma=log_survival_errors_2d, absolute_sigma=True)
    
    slope_2d, intercept_2d = popt_2d
    slope_err_2d, intercept_err_2d = np.sqrt(np.diag(pcov_2d))
    
   
    predicted_2d = linear_func(trimmed_steps_2d, slope_2d, intercept_2d)
    residuals_2d = trimmed_log_survival_counts_2d - predicted_2d
    chisq_2d = np.sum((residuals_2d/log_survival_errors_2d)**2)
    reduced_chisq_2d = chisq_2d / (len(trimmed_steps_2d) - 2)
    
    lambda_estimate_2d = -slope_2d
    lambda_err_2d = slope_err_2d
    
    return (lambda_estimate_2d, lambda_err_2d, trimmed_steps_2d, trimmed_log_survival_counts_2d,
           slope_2d, intercept_2d, slope_err_2d, intercept_err_2d, reduced_chisq_2d, log_survival_errors_2d)


# 2D Calculations
survival_times_2d = [random_walk_2d(J, max_steps_2d) for _ in range(N_walks)]
survival_times_2d_excited = [random_walk_2d(J, max_steps_2d, excited=True) for _ in range(N_walks)]
# Ground State
(lambda_ground_2d, lambda_ground_err_2d, trimmed_steps_2d, trimmed_log_survival_counts_2d,
 slope_2d, intercept_2d, slope_err_2d, intercept_err_2d, red_chi2_ground_2d, log_errors_2d) = calc_arrest_rate_2d(survival_times_2d)
# Excited State
(lambda_excited_2d, lambda_excited_err_2d, steps_2d_excited, log_counts_2d_excited,
 slope_2d_excited, intercept_2d_excited, slope_err_2d_excited, intercept_err_2d_excited,
 red_chi2_2d_excited, log_errors_2d_excited) = calc_arrest_rate_2d(survival_times_2d_excited)

# Compute Energies with errors
E_ground_2d = (h_bar**2 / (m * a**2)) * (lambda_ground_2d * J**2)
E_ground_err_2d = (h_bar**2 / (m * a**2)) * (lambda_ground_err_2d * J**2)
E_excited_2d = (h_bar**2 / (m * a**2)) * (lambda_excited_2d * J**2)
E_excited_err_2d = (h_bar**2 / (m * a**2)) * (lambda_excited_err_2d * J**2)

print("\n2D Results:")
print(f"Ground State Energy: {E_ground_2d:.4f} ± {E_ground_err_2d:.4f}")
print(f"Excited State Energy: {E_excited_2d:.4f} ± {E_excited_err_2d:.4f}")
print(f"Energy Ratio (2D): {E_excited_2d/E_ground_2d:.2f}")
print(f"Ground State Reduced χ²: {red_chi2_ground_2d:.2f}")
print(f"Excited State Reduced χ²: {red_chi2_2d_excited:.2f}")

plt.figure(figsize=(12,8))
plt.errorbar(trimmed_steps_2d, trimmed_log_survival_counts_2d , yerr=log_errors_2d,
             fmt='o', markersize=3, label="2D Ground State Data", color="green")
plt.plot(trimmed_steps_2d, slope_2d * trimmed_steps_2d + intercept_2d,
         linestyle="dashed", color="purple", label="2D Excited State Fit")
plt.xlabel("Number of Steps")
plt.ylabel("Log(Survival Counts)")
plt.title("2D Ground State Random Walk")
plt.legend()
plt.show()
plt.figure(figsize=(12,8))
plt.errorbar(steps_2d_excited, log_counts_2d_excited, yerr=log_errors_2d_excited,
             fmt='o', markersize=3, label="2D Excited State Data", color="purple")
plt.plot(steps_2d_excited, slope_2d_excited * steps_2d_excited + intercept_2d_excited,
         linestyle="dashed", color="purple", label="2D Excited State Fit")
plt.xlabel("Number of Steps")
plt.ylabel("Log(Survival Counts)")
plt.title("2D Excited State Random Walk")
plt.legend()
plt.show()



plot_residual_analysis(step_counts, log_survival_counts, 
                      slope, intercept, "1D Ground State")

plot_residual_analysis(steps_excited, log_counts_excited,
                      slope_excited, intercept_excited, "1D Excited State")

plot_residual_analysis(trimmed_steps_2d, trimmed_log_survival_counts_2d,
                      slope_2d, intercept_2d, "2D Ground State")

plot_residual_analysis(steps_2d_excited, log_counts_2d_excited,
                      slope_2d_excited, intercept_2d_excited, "2D Excited State")

