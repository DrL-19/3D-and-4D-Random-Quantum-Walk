import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats


# Parameters
J = 20
N_walks = 30000
max_steps = 3000
batch_size = 5000


trim_index_3d = 200
trim_index_4d = 200
trim_end_ground_3d = 1500
trim_end_ground_4d = 1500
trim_end_excited_3d = 2500
trim_end_excited_4d = 2500

h_bar = 1.0
m = 1.0
a = 1.0

def linear_func(x, a, b):
    return a * x + b

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def run_single_walk(args):
    walk_func, J, max_steps, excited, min_steps = args
    return walk_func(J, max_steps, excited, min_steps)

def random_walk_3d(J, max_steps, excited=False, min_steps=10):
    if excited:
        # Start at the maximum of first excited state wavefunction
        max_amplitude_radius = 2.08 * J / np.pi
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(1 - 2*np.random.random())
        
        j_1 = int(max_amplitude_radius * np.sin(phi) * np.cos(theta))
        j_2 = int(max_amplitude_radius * np.sin(phi) * np.sin(theta))
        j_3 = int(max_amplitude_radius * np.cos(phi))
    else:
        j_1, j_2, j_3 = 0, 0, 0
    
    for step in range(max_steps):
        move = np.random.randint(0, 6)
        if move == 0: j_1 += 1
        elif move == 1: j_1 -= 1
        elif move == 2: j_2 += 1
        elif move == 3: j_2 -= 1
        elif move == 4: j_3 += 1
        else: j_3 -= 1
        
        if step >= min_steps:
            r_squared = j_1**2 + j_2**2 + j_3**2
            if r_squared >= J**2:
                return step
            if excited and r_squared < (0.5*J)**2:
                return step
    
    return max_steps

def random_walk_4d(J, max_steps, excited=False, min_steps=10):
    if excited:
        r = np.random.uniform(0.65 * J, 0.85 * J)
        direction = np.random.normal(0, 1, 4)
        direction /= np.linalg.norm(direction)  
        coords = r * direction
        j_1, j_2, j_3, j_4 = map(int, coords)
    else:
        j_1, j_2, j_3, j_4 = 0, 0, 0, 0
    
    for step in range(max_steps):
        dim = np.random.randint(0, 4)
        direction = np.random.choice([-1, 1])
        if dim == 0: j_1 += direction
        elif dim == 1: j_2 += direction
        elif dim == 2: j_3 += direction
        else: j_4 += direction
        
        if step >= min_steps:
            r_squared = j_1**2 + j_2**2 + j_3**2 + j_4**2
            if r_squared >= J**2:
                return step
            if excited and r_squared < (0.546*J)**2:
                return step
    
    return max_steps

def calc_arrest_rate(survival_times, max_steps, trim_index, trim_end=None):
    step_counts = np.arange(1, max_steps + 1)
    survival_counts = np.array([np.sum(np.array(survival_times) >= n) for n in step_counts])
    
    
    min_count = 1.0  # minimum count to avoid log(0)
    log_survival_counts = np.log(np.maximum(survival_counts, min_count))
    
   
    if trim_end is None:
        trimmed_steps = step_counts[trim_index:]
        trimmed_log_survival_counts = log_survival_counts[trim_index:]
        survival_counts_array = np.array(survival_counts[trim_index:])
    else:
        trimmed_steps = step_counts[trim_index:-trim_end]
        trimmed_log_survival_counts = log_survival_counts[trim_index:-trim_end]
        survival_counts_array = np.array(survival_counts[trim_index:-trim_end])
    
    
    log_survival_errors = np.where(survival_counts_array > 0,
                                1.0/np.sqrt(survival_counts_array),
                                1.0)  
    

    popt, pcov = curve_fit(linear_func, trimmed_steps, trimmed_log_survival_counts,
                          sigma=log_survival_errors, absolute_sigma=True)
    slope, intercept = popt
    
    residuals = trimmed_log_survival_counts - linear_func(trimmed_steps, slope, intercept)
    
   
    chisq = np.sum((residuals/log_survival_errors)**2)
    reduced_chisq = chisq / (len(trimmed_steps) - 2)
    
    return (float(-slope), float(np.sqrt(pcov[0,0])), trimmed_steps, trimmed_log_survival_counts,
            float(slope), float(intercept), float(reduced_chisq), log_survival_errors, residuals)


def parallel_random_walks(walk_func, J, max_steps, N_walks, excited=False, min_steps=5, 
                        n_processes=None, batch_size=5000):
    """Improved parallel processing with batch processing"""
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)
    
    results = []
    n_batches = (N_walks + batch_size - 1) // batch_size
    
    for batch in range(n_batches):
        current_batch_size = min(batch_size, N_walks - batch * batch_size)
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            args_list = [(walk_func, J, max_steps, excited, min_steps) 
                        for _ in range(current_batch_size)]
            batch_results = list(tqdm(
                executor.map(run_single_walk, args_list),
                total=current_batch_size,
                desc=f"Batch {batch + 1}/{n_batches}"
            ))
            results.extend(batch_results)
    
    return results

def plot_residual_analysis(steps, log_counts, params, residuals, errors, title_prefix):
    """Plot residual analysis including Q-Q plot"""
    
    slope, intercept = params 
    
    
    predicted = linear_func(steps, slope, intercept)
    residuals = log_counts - predicted
    std_dev = np.std(residuals)
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix} Residual Analysis')
    
    # 1. Residuals vs Steps plot
    ax1.scatter(steps, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Steps')
    ax1.set_ylim([-3*std_dev, 3*std_dev])
    
    # 2. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    # 3. Histogram with normal distribution
    n_bins = min(50, len(residuals)//3)  # Increased bin count
    counts, bins, _ = ax3.hist(residuals, bins=n_bins, density=True, alpha=0.7)
   
    x = np.linspace(-4*std_dev, 4*std_dev, 200)
    ax3.plot(x, stats.norm.pdf(x, np.mean(residuals), std_dev), 
             'r-', lw=2, label=f'Normal\nσ={std_dev:.3f}')
    
    ax3.set_xlim(-4*std_dev, 4*std_dev)
    ax3.set_ylim(0, 1.1*max(counts.max(), stats.norm.pdf(0, 0, std_dev)))
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Histogram of Residuals')
    ax3.legend()
    
    # 4. Standard deviation text box
    standardized_residuals = (residuals - np.mean(residuals)) / std_dev
    D, p_value = stats.kstest(standardized_residuals, 'norm')
    
    test_text = f"""
    Standard Deviation: {std_dev:.4f}
    KS Test:
      D-statistic: {D:.4f}
      p-value: {p_value:.4f}
    """
    ax4.text(0.5, 0.5, test_text, ha='center', va='center', 
             transform=ax4.transAxes, fontsize=11)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_energy(lambda_val, J, dimension, h_bar=1.0, m=1.0, a=1.0):
    """Calculate energy with proper dimensional scaling and correction"""
    # Theoretical factors
    if dimension == 3:
    
        theoretical_factor = (np.pi)**2 / (4 * J**2)
        energy_scale = (h_bar**2 / (2*m)) * (np.pi/a)**2
        correction_factor = 1
    elif dimension == 4:
        
        theoretical_factor = (3.8317)**2 / (4 * J**2)
        energy_scale = (h_bar**2 / (2*m)) * (3.8317/a)**2
        correction_factor = 1
    return energy_scale * (lambda_val/theoretical_factor) * correction_factor

def run_simulation():
    """Main simulation function"""
    # Run 3D simulations
    print("\nRunning 3D simulations...")
    survival_times_3d = parallel_random_walks(random_walk_3d, J, max_steps, N_walks, 
                                            batch_size=batch_size)
    
    # Ground state
    (lambda_ground_3d, lambda_err_3d, steps_3d, log_counts_3d,
     slope_3d, intercept_3d, red_chi2_3d, log_errors_3d, residuals_3d) = calc_arrest_rate(
        survival_times_3d, max_steps, trim_index_3d, trim_end_ground_3d)  # Changed to trim_end_ground_3d
    
    survival_times_3d_excited = parallel_random_walks(random_walk_3d, J, max_steps, N_walks, excited=True, batch_size=batch_size)
    # Excited state
    (lambda_excited_3d, lambda_err_excited_3d, steps_3d_ex, log_counts_3d_ex,
     slope_3d_ex, intercept_3d_ex, red_chi2_excited_3d, log_errors_3d_ex, residuals_3d_ex) = calc_arrest_rate(
        survival_times_3d_excited, max_steps, trim_index_3d, trim_end_excited_3d)  # Changed to trim_end_excited_3d

    # Run 4D simulations
    print("\nRunning 4D simulations...")
    survival_times_4d = parallel_random_walks(random_walk_4d, J, max_steps, N_walks,
                                            batch_size=batch_size)
    
    # Ground state
    (lambda_ground_4d, lambda_err_4d, steps_4d, log_counts_4d,
     slope_4d, intercept_4d, red_chi2_4d, log_errors_4d, residuals_4d) = calc_arrest_rate(
        survival_times_4d, max_steps, trim_index_4d, trim_end_ground_4d)  # Changed to trim_end_ground_4d
    
    survival_times_4d_excited = parallel_random_walks(random_walk_4d, J, max_steps, N_walks, excited=True, batch_size=batch_size)
    # Excited state 
    (lambda_excited_4d, lambda_err_excited_4d, steps_4d_ex, log_counts_4d_ex,
     slope_4d_ex, intercept_4d_ex, red_chi2_excited_4d, log_errors_4d_ex, residuals_4d_ex) = calc_arrest_rate(
        survival_times_4d_excited, max_steps, trim_index_4d, trim_end_excited_4d)  # Changed to trim_end_excited_4d
   
    # Theoretical Energy Values
    theoretical_E_3d = (h_bar**2 / (2*m)) * (3.141592654/a)**2
    theoretical_E_4d = (h_bar**2 / (2*m)) * (3.831705970/a)**2
    theoretical_E_3d_excited = (h_bar**2 / (2*m)) * (4.493409458/a)**2
    theoretical_E_4d_excited = (h_bar**2 / (2*m)) * (7.01558667/a)**2   
    # Calculate energies
    E_ground_3d = calculate_energy(lambda_ground_3d, J, 3)
    E_excited_3d = calculate_energy(lambda_excited_3d, J, 3)
    E_ground_4d = calculate_energy(lambda_ground_4d, J, 4)
    E_excited_4d = calculate_energy(lambda_excited_4d, J, 4)

    # Error calculations
    E_ground_3d_err = calculate_energy(lambda_err_3d, J, 3)
    E_excited_3d_err = calculate_energy(lambda_err_excited_3d, J, 3)
    E_ground_4d_err = calculate_energy(lambda_err_4d, J, 4)
    E_excited_4d_err = calculate_energy(lambda_err_excited_4d, J, 4)

    # Print results
    print("\n=== 3D Results ===")
    print(f"Ground State Energy: {E_ground_3d:.4f} ± {E_ground_3d_err:.4f}, Theory: {theoretical_E_3d:.4f}")
    print(f"Excited State Energy: {E_excited_3d:.4f} ± {E_excited_3d_err:.4f}, Theory: {theoretical_E_3d_excited:.4f}")
    print(f"Ground State Reduced χ²: {red_chi2_3d:.2f} ")
    print(f"Excited State Reduced χ²: {red_chi2_excited_3d:.2f}")

    print("\n=== 4D Results ===")
    print(f"Ground State Energy: {E_ground_4d:.4f} ± {E_ground_4d_err:.4f}, Theory: {theoretical_E_4d:.4f}")
    print(f"Excited State Energy: {E_excited_4d:.4f} ± {E_excited_4d_err:.4f}, Theory: {theoretical_E_4d_excited:.4f}")
    print(f"Ground State Reduced χ²: {red_chi2_4d:.2f}")
    print(f"Excited State Reduced χ²: {red_chi2_excited_4d:.2f}")

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Random Walk Survival Analysis', fontsize=16)
    
    # 3D Ground State
    axs[0,0].errorbar(steps_3d, log_counts_3d, yerr=log_errors_3d, fmt='o', markersize=2, 
                      errorevery=15, label="Data", color='blue')
    axs[0,0].plot(steps_3d, slope_3d*steps_3d + intercept_3d, 'b--', label="Linear Fit")
    axs[0,0].set_title('3D Ground State')
    axs[0,0].set_xlabel('Number of Steps')
    axs[0,0].set_ylabel('Log(Survival Counts)')
    axs[0,0].legend()
    axs[0,0].grid(True)

    # 3D Excited State
    axs[0,1].errorbar(steps_3d_ex, log_counts_3d_ex, yerr=log_errors_3d_ex, fmt='o', markersize=2,
                      errorevery=15, label="Data", color='red')
    axs[0,1].plot(steps_3d_ex, slope_3d_ex*steps_3d_ex + intercept_3d_ex, 'r--', label="Linear Fit")
    axs[0,1].set_title('3D Excited State')
    axs[0,1].set_xlabel('Number of Steps')
    axs[0,1].set_ylabel('Log(Survival Counts)')
    axs[0,1].legend()
    axs[0,1].grid(True)
    
    # 4D Ground State
    axs[1,0].errorbar(steps_4d, log_counts_4d, yerr=log_errors_4d, fmt='o', markersize=2, 
                      errorevery=15, label="Data", color='green')
    axs[1,0].plot(steps_4d, slope_4d*steps_4d + intercept_4d, 'g--', label="Linear Fit")
    axs[1,0].set_title('4D Ground State')
    axs[1,0].set_xlabel('Number of Steps')
    axs[1,0].set_ylabel('Log(Survival Counts)')
    axs[1,0].legend()
    axs[1,0].grid(True)
    
    # 4D Excited State
    axs[1,1].errorbar(steps_4d_ex, log_counts_4d_ex, yerr=log_errors_4d_ex, fmt='o', markersize=2,
                      errorevery=15, label="Data", color='purple')
    axs[1,1].plot(steps_4d_ex, slope_4d_ex*steps_4d_ex + intercept_4d_ex, 'm--', label="Linear Fit")
    axs[1,1].set_title('4D Excited State')
    axs[1,1].set_xlabel('Number of Steps')
    axs[1,1].set_ylabel('Log(Survival Counts)')
    axs[1,1].legend()
    axs[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Residual analysis plots 
    plot_residual_analysis(steps_3d, log_counts_3d, [slope_3d, intercept_3d], residuals_3d, log_errors_3d, "3D Ground State")
    plot_residual_analysis(steps_3d_ex, log_counts_3d_ex, [slope_3d_ex, intercept_3d_ex], residuals_3d_ex, log_errors_3d_ex, "3D Excited State")
    plot_residual_analysis(steps_4d, log_counts_4d, [slope_4d, intercept_4d], residuals_4d, log_errors_4d, "4D Ground State")
    plot_residual_analysis(steps_4d_ex, log_counts_4d_ex, [slope_4d_ex, intercept_4d_ex], residuals_4d_ex, log_errors_4d_ex, "4D Excited State")
    
if __name__ == "__main__":
    run_simulation()
   
